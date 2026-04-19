import csv
import hashlib
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, create_engine, func
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from app.observability import get_trace_id, setup_telemetry

SERVICE_NAME = os.getenv("SERVICE_NAME", "payment-service")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./payment.db")
SEED_DATA_DIR = Path(os.getenv("SEED_DATA_DIR", ""))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(SERVICE_NAME)
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["service", "path", "method", "status_code"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "Request latency", ["service", "path", "method"])
PAYMENTS_FAILED = Counter("payments_failed_total", "Failed payments", ["service"])


class Base(DeclarativeBase):
    pass


class Payment(Base):
    __tablename__ = "payments"
    payment_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int] = mapped_column(Integer)
    amount: Mapped[float] = mapped_column(Float)
    method: Mapped[str] = mapped_column(String(30))
    status: Mapped[str] = mapped_column(String(30))
    reference: Mapped[str] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime)


class IdempotencyKey(Base):
    __tablename__ = "idempotency_keys"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(120), unique=True)
    request_hash: Mapped[str] = mapped_column(String(64))
    payment_id: Mapped[Optional[int]] = mapped_column(ForeignKey("payments.payment_id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    payment: Mapped[Optional[Payment]] = relationship()


engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class ApiError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400):
        self.code = code
        self.message = message
        self.status_code = status_code


class ChargeRequest(BaseModel):
    order_id: int
    amount: float
    method: str
    reference: Optional[str] = None
    simulate_failure: bool = False


class RefundRequest(BaseModel):
    reason: Optional[str] = None


class PaymentOut(BaseModel):
    payment_id: int
    order_id: int
    amount: float
    method: str
    status: str
    reference: str
    created_at: datetime

    class Config:
        from_attributes = True


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def next_id(db: Session, model, column) -> int:
    return (db.query(func.max(column)).scalar() or 0) + 1


def parse_dt(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def seed_data():
    if not SEED_DATA_DIR or not SEED_DATA_DIR.exists():
        return
    with SessionLocal() as db:
        if db.query(Payment).first():
            return
        with (SEED_DATA_DIR / "ofd_payments.csv").open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                db.add(Payment(payment_id=int(row["payment_id"]), order_id=int(row["order_id"]), amount=float(row["amount"]), method=row["method"], status=row["status"], reference=row["reference"], created_at=parse_dt(row["created_at"])))
        db.commit()


def request_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    seed_data()
    yield


app = FastAPI(title="Payment Service", version="1.0.0", lifespan=lifespan)
setup_telemetry(app, engine, SERVICE_NAME)


@app.middleware("http")
async def telemetry(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-Id", str(uuid.uuid4()))
    request.state.correlation_id = correlation_id
    started = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - started
    REQUEST_COUNT.labels(SERVICE_NAME, request.url.path, request.method, str(response.status_code)).inc()
    REQUEST_LATENCY.labels(SERVICE_NAME, request.url.path, request.method).observe(elapsed)
    response.headers["X-Correlation-Id"] = correlation_id
    logger.info(json.dumps({"service": SERVICE_NAME, "correlationId": correlation_id, "traceId": get_trace_id(), "path": request.url.path, "statusCode": response.status_code, "latencyMs": round(elapsed * 1000, 2)}))
    return response


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError):
    return JSONResponse(status_code=exc.status_code, content={"code": exc.code, "message": exc.message, "correlationId": getattr(request.state, "correlation_id", str(uuid.uuid4()))})


@app.get("/health")
def health():
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/payments", response_model=list[PaymentOut])
def list_payments(status: Optional[str] = None, method: Optional[str] = None, limit: int = Query(10, ge=1, le=100), offset: int = Query(0, ge=0), db: Session = Depends(get_db)):
    query = db.query(Payment)
    if status:
        query = query.filter(Payment.status == status)
    if method:
        query = query.filter(Payment.method == method)
    return query.order_by(Payment.payment_id.desc()).offset(offset).limit(limit).all()


@app.get("/v1/payments/{payment_id}", response_model=PaymentOut)
def get_payment(payment_id: int, db: Session = Depends(get_db)):
    payment = db.get(Payment, payment_id)
    if not payment:
        raise ApiError("PAYMENT_NOT_FOUND", f"Payment {payment_id} not found", 404)
    return payment


@app.post("/v1/payments/charge")
def charge_payment(payload: ChargeRequest, db: Session = Depends(get_db), idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key")):
    if not idempotency_key:
        raise ApiError("MISSING_IDEMPOTENCY_KEY", "Idempotency-Key header is required", 400)
    payload_hash = request_hash(payload.model_dump())
    existing = db.query(IdempotencyKey).filter(IdempotencyKey.idempotency_key == idempotency_key).first()
    if existing:
        if existing.request_hash != payload_hash:
            raise ApiError("IDEMPOTENCY_CONFLICT", "Same Idempotency-Key used with different payload", 409)
        payment = db.get(Payment, existing.payment_id)
        return {"payment_id": payment.payment_id, "order_id": payment.order_id, "status": payment.status, "amount": payment.amount, "method": payment.method, "reference": payment.reference, "idempotentReplay": True}

    method = payload.method.upper()
    if method not in {"CARD", "UPI", "WALLET", "COD"}:
        raise ApiError("UNSUPPORTED_PAYMENT_METHOD", f"Payment method {payload.method} is not supported", 400)
    status = "PENDING" if method == "COD" else "FAILED" if payload.simulate_failure else "SUCCESS"
    payment = Payment(payment_id=next_id(db, Payment, Payment.payment_id), order_id=payload.order_id, amount=round(payload.amount, 2), method=method, status=status, reference=payload.reference or f"PAY-{uuid.uuid4().hex[:10].upper()}", created_at=datetime.utcnow())
    db.add(payment)
    db.flush()
    db.add(IdempotencyKey(id=next_id(db, IdempotencyKey, IdempotencyKey.id), idempotency_key=idempotency_key, request_hash=payload_hash, payment_id=payment.payment_id, created_at=datetime.utcnow()))
    db.commit()
    if status == "FAILED":
        PAYMENTS_FAILED.labels(SERVICE_NAME).inc()
    return {"payment_id": payment.payment_id, "order_id": payment.order_id, "status": payment.status, "amount": payment.amount, "method": payment.method, "reference": payment.reference, "idempotentReplay": False}


@app.post("/v1/payments/{payment_id}/refund")
def refund_payment(payment_id: int, request: RefundRequest, db: Session = Depends(get_db)):
    payment = db.get(Payment, payment_id)
    if not payment:
        raise ApiError("PAYMENT_NOT_FOUND", f"Payment {payment_id} not found", 404)
    if payment.status == "FAILED":
        raise ApiError("REFUND_NOT_ALLOWED", "Failed payments cannot be refunded", 409)
    payment.status = "REFUNDED"
    db.commit()
    return {"payment_id": payment.payment_id, "status": payment.status, "reason": request.reason}
