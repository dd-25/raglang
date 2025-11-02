import logging
from fastapi import FastAPI
from beetu_v2.config import settings
from beetu_v2.routes import health, upload, query
from beetu_v2.constants import APP_SETTINGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

def create_app() -> FastAPI:
    app = FastAPI(
        title=APP_SETTINGS.APP_NAME,
        version=APP_SETTINGS.VERSION,
        description=APP_SETTINGS.DESCRIPTION
    )
    
    @app.on_event("startup")
    async def startup_event():
        """Validate configuration on startup"""
        from beetu_v2.config import validate_required_keys
        try:
            validate_required_keys()
            print("✅ Configuration validation passed")
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            raise
    
    @app.on_event("shutdown") 
    async def shutdown_event():
        """Cleanup resources on shutdown"""
        print("🔄 Shutting down gracefully...")
    
    @app.get("/")
    async def root():
        return {f"message": f"Welcome to {APP_SETTINGS.APP_NAME}"}
    
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(upload.router, prefix="/upload", tags=["Upload"])
    app.include_router(query.router, prefix="/query", tags=["Query"])

    return app


app = create_app()

def main():
    import uvicorn

    uvicorn.run(
        "beetu_v2.main:app",
        host="0.0.0.0",
        port=settings.APP_PORT,
        reload=True
    )
    
if __name__ == "__main__":
    main()