# esign_routes.py - DocuSign E-Signature Integration
"""
Send contracts for electronic signature via DocuSign.

Flow: Generate PDF → Create Envelope → Get Signing URL → Webhook Status Update

Endpoints:
- POST /{id}/esign/send           — Create DocuSign envelope and send for signing
- GET  /{id}/esign/signing-url    — Get embedded signing URL for iframe/redirect
- GET  /{id}/esign/status         — Check envelope status
- POST /api/esign/webhook         — DocuSign Connect webhook callback

Requires DocuSign developer account credentials in .env:
  DOCUSIGN_INTEGRATION_KEY, DOCUSIGN_USER_ID, DOCUSIGN_ACCOUNT_ID,
  DOCUSIGN_RSA_PRIVATE_KEY (base64 encoded)
"""

import os
import base64
import json
from io import BytesIO
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from config import DB_CONFIG
import psycopg2
from psycopg2.extras import RealDictCursor

router = APIRouter(tags=["e-signature"])


# ==================== DOCUSIGN CONFIG ====================

DOCUSIGN_CONFIG = {
    "integration_key": os.getenv("DOCUSIGN_INTEGRATION_KEY", ""),
    "user_id": os.getenv("DOCUSIGN_USER_ID", ""),
    "account_id": os.getenv("DOCUSIGN_ACCOUNT_ID", ""),
    "private_key": os.getenv("DOCUSIGN_RSA_PRIVATE_KEY", ""),  # Base64 encoded
    "base_url": os.getenv("DOCUSIGN_BASE_URL", "https://demo.docusign.net/restapi"),
    "oauth_host": os.getenv("DOCUSIGN_OAUTH_HOST", "account-d.docusign.com"),
    "return_url": os.getenv("DOCUSIGN_RETURN_URL", "http://localhost:3000/signing-complete"),
}


def _is_configured() -> bool:
    """Check if DocuSign credentials are set."""
    return bool(
        DOCUSIGN_CONFIG["integration_key"]
        and DOCUSIGN_CONFIG["user_id"]
        and DOCUSIGN_CONFIG["account_id"]
    )


def _get_docusign_client():
    """Create and authenticate a DocuSign API client."""
    if not _is_configured():
        raise HTTPException(
            status_code=503,
            detail={
                "message": "DocuSign is not configured",
                "hint": "Add DOCUSIGN_INTEGRATION_KEY, DOCUSIGN_USER_ID, "
                        "DOCUSIGN_ACCOUNT_ID, and DOCUSIGN_RSA_PRIVATE_KEY to .env",
                "docs": "https://developers.docusign.com/docs/esign-rest-api/quickstart",
            }
        )

    try:
        from docusign_esign import ApiClient
        from docusign_esign.client.api_exception import ApiException

        client = ApiClient()
        client.set_oauth_host_name(DOCUSIGN_CONFIG["oauth_host"])

        # Resolve private key — supports 3 formats:
        #   1. File path: /path/to/private.pem
        #   2. Base64-encoded full PEM (single line in .env)
        #   3. PEM content with literal \n (e.g. "-----BEGIN...-----\nMIIE...\n-----END...-----")
        raw_key = DOCUSIGN_CONFIG["private_key"]
        if not raw_key:
            raise HTTPException(
                status_code=503,
                detail="DOCUSIGN_RSA_PRIVATE_KEY not set in .env"
            )

        private_key_bytes = None

        # Format 1: File path
        if os.path.isfile(raw_key):
            with open(raw_key, "rb") as f:
                private_key_bytes = f.read()

        # Format 3: PEM with literal \n (from .env)
        elif "-----BEGIN" in raw_key:
            private_key_bytes = raw_key.replace("\\n", "\n").encode()

        # Format 2: Base64-encoded full PEM
        else:
            try:
                decoded = base64.b64decode(raw_key)
                # Check if it decoded into a valid PEM
                if b"-----BEGIN" in decoded:
                    private_key_bytes = decoded
                else:
                    # Raw base64 might be the key body — wrap it in PEM headers
                    private_key_bytes = (
                        b"-----BEGIN RSA PRIVATE KEY-----\n"
                        + decoded
                        + b"\n-----END RSA PRIVATE KEY-----\n"
                    )
            except Exception:
                private_key_bytes = raw_key.encode()

        # Validate key length (RSA private key PEM is typically 1600+ bytes)
        if len(private_key_bytes) < 500:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": f"Private key appears truncated ({len(private_key_bytes)} bytes). "
                               "An RSA private key is typically 1600+ bytes.",
                    "hint": "Option 1: Save the key as a file and set DOCUSIGN_RSA_PRIVATE_KEY=/path/to/private.pem\n"
                            "Option 2: Base64 encode the ENTIRE PEM file: "
                            "cat private.pem | base64 | tr -d '\\n' > key.b64\n"
                            "Option 3: Use literal \\n: "
                            'DOCUSIGN_RSA_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----\\nMIIE...\\n-----END RSA PRIVATE KEY-----"',
                }
            )

        # Get JWT token
        token_response = client.request_jwt_user_token(
            client_id=DOCUSIGN_CONFIG["integration_key"],
            user_id=DOCUSIGN_CONFIG["user_id"],
            oauth_host_name=DOCUSIGN_CONFIG["oauth_host"],
            private_key_bytes=private_key_bytes,
            expires_in=3600,
            scopes=["signature", "impersonation"],
        )

        # IMPORTANT: The JWT auth flow corrupts the auth client's host.
        # Create a NEW client for API operations with the correct base URI.
        base_uri = "https://demo.docusign.net"
        try:
            user_info = client.get_user_info(token_response.access_token)
            accounts = user_info.get_accounts()
            for acc in accounts:
                if acc.account_id == DOCUSIGN_CONFIG["account_id"]:
                    base_uri = acc.base_uri
                    break
            else:
                if accounts:
                    base_uri = accounts[0].base_uri
        except Exception:
            pass

        api_client = ApiClient()
        api_client.host = f"{base_uri}/restapi"
        api_client.set_default_header(
            "Authorization", f"Bearer {token_response.access_token}"
        )

        return api_client

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="docusign-esign not installed. Run: pip install docusign-esign"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"DocuSign authentication failed: {str(e)}"
        )


# ==================== TABLE SETUP ====================

_columns_added = False

def _ensure_columns():
    """Add e-signature columns to contracts table if they don't exist."""
    global _columns_added
    if _columns_added:
        return

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                ALTER TABLE contracts ADD COLUMN IF NOT EXISTS envelope_id TEXT;
                ALTER TABLE contracts ADD COLUMN IF NOT EXISTS signing_status TEXT DEFAULT 'not_sent';
                ALTER TABLE contracts ADD COLUMN IF NOT EXISTS signed_at TIMESTAMPTZ;
                ALTER TABLE contracts ADD COLUMN IF NOT EXISTS signer_email TEXT;
                ALTER TABLE contracts ADD COLUMN IF NOT EXISTS signer_name TEXT;
            """)
            conn.commit()
        _columns_added = True
    finally:
        conn.close()


# ==================== PYDANTIC MODELS ====================

class SendForSigningRequest(BaseModel):
    signer_email: str = Field(..., description="Email of the signer")
    signer_name: str = Field(..., description="Name of the signer")
    email_subject: Optional[str] = Field(None, description="Custom email subject")
    email_body: Optional[str] = Field(None, description="Custom email body")


class SigningStatusResponse(BaseModel):
    contract_id: str
    envelope_id: Optional[str]
    signing_status: str
    signer_email: Optional[str]
    signer_name: Optional[str]
    signed_at: Optional[datetime]


# ==================== HELPERS ====================

def get_db():
    return psycopg2.connect(**DB_CONFIG)


def _get_contract_pdf(contract_id: str) -> bytes:
    """Generate PDF for the contract using export_routes logic."""
    from export_routes import _get_contract_data, _build_pdf
    data = _get_contract_data(contract_id)
    buf = _build_pdf(data)
    return buf.getvalue()


def _update_signing_status(
    contract_id: str = None,
    envelope_id: str = None,
    status: str = None,
    signer_email: str = None,
    signer_name: str = None,
    signed_at: datetime = None,
):
    """Update signing status in the contracts table."""
    conn = get_db()
    try:
        with conn.cursor() as cur:
            updates = []
            params = []

            if status:
                updates.append("signing_status = %s")
                params.append(status)
            if envelope_id:
                updates.append("envelope_id = %s")
                params.append(envelope_id)
            if signer_email:
                updates.append("signer_email = %s")
                params.append(signer_email)
            if signer_name:
                updates.append("signer_name = %s")
                params.append(signer_name)
            if signed_at:
                updates.append("signed_at = %s")
                params.append(signed_at)

            updates.append("updated_at = NOW()")

            if contract_id:
                where = "id = %s"
                params.append(contract_id)
            elif envelope_id:
                where = "envelope_id = %s"
                params.append(envelope_id)
            else:
                return

            sql = f"UPDATE contracts SET {', '.join(updates)} WHERE {where}"
            cur.execute(sql, params)
            conn.commit()
    finally:
        conn.close()


# ==================== ROUTES ====================

@router.post("/api/contracts/{contract_id}/esign/send")
async def send_for_signing(contract_id: str, request: SendForSigningRequest):
    """
    Create a DocuSign envelope and send the contract for e-signature.

    1. Generates the contract PDF
    2. Creates a DocuSign envelope with the PDF
    3. Sends to the specified signer via email
    4. Stores the envelope_id in the contracts table
    """
    _ensure_columns()

    # Check contract exists and isn't already sent
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, title, signing_status, envelope_id
                FROM contracts WHERE id = %s
            """, (contract_id,))
            contract = cur.fetchone()
    finally:
        conn.close()

    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")

    if contract.get("signing_status") in ("sent", "delivered", "signed"):
        raise HTTPException(
            status_code=409,
            detail=f"Contract already has signing status: {contract['signing_status']}. "
                   f"Envelope: {contract.get('envelope_id')}"
        )

    # Generate PDF
    try:
        pdf_bytes = _get_contract_pdf(contract_id)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate contract PDF: {str(e)}"
        )

    # Create DocuSign envelope
    client = _get_docusign_client()

    try:
        from docusign_esign import EnvelopesApi, EnvelopeDefinition, Document
        from docusign_esign import Signer, SignHere, Tabs, Recipients

        # Create document from PDF
        doc_b64 = base64.b64encode(pdf_bytes).decode("ascii")
        document = Document(
            document_base64=doc_b64,
            name=contract["title"] or "Contract",
            file_extension="pdf",
            document_id="1",
        )

        # Define signer with sign-here tab
        sign_here = SignHere(
            anchor_string="/sig1/",
            anchor_units="pixels",
            anchor_y_offset="10",
            anchor_x_offset="20",
        )

        signer = Signer(
            email=request.signer_email,
            name=request.signer_name,
            recipient_id="1",
            routing_order="1",
            tabs=Tabs(sign_here_tabs=[sign_here]),
        )

        # Create envelope
        envelope_definition = EnvelopeDefinition(
            email_subject=request.email_subject or f"Please sign: {contract['title']}",
            email_blurb=request.email_body or "Please review and sign this contract.",
            documents=[document],
            recipients=Recipients(signers=[signer]),
            status="sent",
        )

        envelopes_api = EnvelopesApi(client)
        result = envelopes_api.create_envelope(
            account_id=DOCUSIGN_CONFIG["account_id"],
            envelope_definition=envelope_definition,
        )

        envelope_id = result.envelope_id

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"DocuSign envelope creation failed: {str(e)}"
        )

    # Update contract with envelope info
    _update_signing_status(
        contract_id=contract_id,
        envelope_id=envelope_id,
        status="sent",
        signer_email=request.signer_email,
        signer_name=request.signer_name,
    )

    # Auto-version: snapshot after sending for signature
    try:
        from version_routes import create_version_snapshot
        create_version_snapshot(
            contract_id,
            f"Sent for e-signature to {request.signer_email}"
        )
    except Exception:
        pass

    return {
        "message": "Contract sent for e-signature",
        "contract_id": contract_id,
        "envelope_id": envelope_id,
        "signer_email": request.signer_email,
        "signer_name": request.signer_name,
        "status": "sent",
    }


@router.get("/api/contracts/{contract_id}/esign/signing-url")
async def get_signing_url(contract_id: str):
    """
    Get an embedded signing URL for the signer.
    Use this to open DocuSign signing in an iframe or redirect.
    URL is valid for 5 minutes.
    """
    _ensure_columns()

    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, envelope_id, signing_status, signer_email, signer_name
                FROM contracts WHERE id = %s
            """, (contract_id,))
            contract = cur.fetchone()
    finally:
        conn.close()

    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")

    if not contract.get("envelope_id"):
        raise HTTPException(
            status_code=400,
            detail="Contract has not been sent for signing yet. Use POST /esign/send first."
        )

    if contract.get("signing_status") == "signed":
        raise HTTPException(
            status_code=409,
            detail="Contract is already signed"
        )

    client = _get_docusign_client()

    try:
        from docusign_esign import EnvelopesApi, RecipientViewRequest

        view_request = RecipientViewRequest(
            return_url=DOCUSIGN_CONFIG["return_url"],
            authentication_method="none",
            email=contract["signer_email"],
            user_name=contract["signer_name"],
            client_user_id="1",  # Required for embedded signing
        )

        envelopes_api = EnvelopesApi(client)
        result = envelopes_api.create_recipient_view(
            account_id=DOCUSIGN_CONFIG["account_id"],
            envelope_id=contract["envelope_id"],
            recipient_view_request=view_request,
        )

        return {
            "signing_url": result.url,
            "expires_in_minutes": 5,
            "contract_id": contract_id,
            "envelope_id": contract["envelope_id"],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get signing URL: {str(e)}"
        )


@router.get("/api/contracts/{contract_id}/esign/status", response_model=SigningStatusResponse)
async def get_signing_status(contract_id: str):
    """
    Get the current e-signature status for a contract.

    If DocuSign is configured and an envelope exists, also queries
    DocuSign for the latest status and updates the local DB.
    """
    _ensure_columns()

    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, envelope_id, signing_status, signer_email,
                       signer_name, signed_at
                FROM contracts WHERE id = %s
            """, (contract_id,))
            contract = cur.fetchone()
    finally:
        conn.close()

    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")

    # If we have an envelope and DocuSign is configured, get live status
    if contract.get("envelope_id") and _is_configured():
        try:
            client = _get_docusign_client()
            from docusign_esign import EnvelopesApi

            envelopes_api = EnvelopesApi(client)
            envelope = envelopes_api.get_envelope(
                account_id=DOCUSIGN_CONFIG["account_id"],
                envelope_id=contract["envelope_id"],
            )

            ds_status = envelope.status  # sent, delivered, completed, declined, voided
            local_status = _map_docusign_status(ds_status)

            # Update local DB if status changed
            if local_status != contract.get("signing_status"):
                signed_at = None
                if local_status == "signed":
                    signed_at = datetime.now()
                _update_signing_status(
                    contract_id=contract_id,
                    status=local_status,
                    signed_at=signed_at,
                )
                contract["signing_status"] = local_status
                contract["signed_at"] = signed_at

        except Exception:
            pass  # Fall through to return cached status

    return {
        "contract_id": str(contract["id"]),
        "envelope_id": contract.get("envelope_id"),
        "signing_status": contract.get("signing_status", "not_sent"),
        "signer_email": contract.get("signer_email"),
        "signer_name": contract.get("signer_name"),
        "signed_at": contract.get("signed_at"),
    }


@router.post("/api/esign/webhook")
async def docusign_webhook(request: Request):
    """
    DocuSign Connect webhook callback.

    DocuSign sends POST notifications when envelope status changes.
    Configure this URL in DocuSign Admin → Connect → Add Configuration.

    Webhook URL: https://your-domain.com/api/esign/webhook
    """
    try:
        body = await request.json()
    except Exception:
        # DocuSign may also send XML; handle gracefully
        body_bytes = await request.body()
        return {"message": "Received", "format": "non-json", "size": len(body_bytes)}

    # Extract envelope info from DocuSign webhook payload
    envelope_id = body.get("envelopeId") or body.get("envelope_id")
    status = body.get("status") or body.get("envelope_status")

    if not envelope_id:
        return {"message": "No envelope ID in payload"}

    local_status = _map_docusign_status(status)

    signed_at = None
    if local_status == "signed":
        signed_at = datetime.now()

    _update_signing_status(
        envelope_id=envelope_id,
        status=local_status,
        signed_at=signed_at,
    )

    return {
        "message": "Status updated",
        "envelope_id": envelope_id,
        "status": local_status,
    }


def _map_docusign_status(ds_status: str) -> str:
    """Map DocuSign envelope status to local status."""
    if not ds_status:
        return "not_sent"
    status_map = {
        "created": "draft",
        "sent": "sent",
        "delivered": "delivered",
        "completed": "signed",
        "signed": "signed",
        "declined": "declined",
        "voided": "voided",
    }
    return status_map.get(ds_status.lower(), ds_status.lower())
