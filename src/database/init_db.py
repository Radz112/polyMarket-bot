import asyncio
from src.database.postgres import DatabaseManager
from src.database.models import Base

async def init_models():
    db = DatabaseManager()
    await db.connect() # This creates the engine
    
    print("Creating tables...")
    async with db._engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Tables created successfully.")
    
    await db.disconnect()

if __name__ == "__main__":
    asyncio.run(init_models())
