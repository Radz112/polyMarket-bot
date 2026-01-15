import pytest
import asyncio
import json
import aiohttp
from aiohttp import web
from src.api.ws_client import ClobWsClient
import logging

# Set up logging for tests if needed
logging.basicConfig(level=logging.INFO)

class MockWsServer:
    def __init__(self, port=0):
        self.messages_received = []
        self.app = web.Application()
        self.app.router.add_get('/ws/market', self.ws_handler)
        self.runner = None
        self.ws = None
        self.port = port
        self.site = None

    async def ws_handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.ws = ws
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    self.messages_received.append(data)
                    if data.get("operation") == "subscribe":
                        await ws.send_json({"type": "subscription_success"})
                    elif data.get("operation") == "unsubscribe":
                        await ws.send_json({"type": "unsubscription_success"})
                except json.JSONDecodeError:
                    pass
            elif msg.type == web.WSMsgType.CLOSE:
                await ws.close()
        return ws

    async def start(self):
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, '127.0.0.1', self.port)
        await self.site.start()
        # Update port to the actual one assigned if it was 0
        if self.port == 0:
            self.port = self.runner.addresses[0][1]

    async def stop(self):
        if self.runner:
            if self.ws and not self.ws.closed:
                await self.ws.close()
            await self.runner.cleanup()
            self.runner = None
            self.ws = None
            self.site = None

import pytest_asyncio

@pytest_asyncio.fixture
async def mock_server():
    server = MockWsServer()
    await server.start()
    yield server
    await server.stop()

@pytest.mark.asyncio
async def test_ws_subscription(mock_server):
    url = f"ws://127.0.0.1:{mock_server.port}/ws/market"
    client = ClobWsClient(url=url, asset_ids=["token1"])
    
    task = asyncio.create_task(client.connect())
    
    # Wait for subscription
    for _ in range(20):
        if any(m.get("operation") == "subscribe" for m in mock_server.messages_received):
            break
        await asyncio.sleep(0.1)
    
    assert any(m.get("operation") == "subscribe" for m in mock_server.messages_received)
    
    await client.stop()
    task.cancel()

@pytest.mark.asyncio
async def test_ws_callback_resilience(mock_server):
    url = f"ws://127.0.0.1:{mock_server.port}/ws/market"
    client = ClobWsClient(url=url)
    
    results = []
    def faulty_callback(data):
        raise ValueError("Boom")
    
    def healthy_callback(data):
        results.append(data)

    client.register_callback(faulty_callback)
    client.register_callback(healthy_callback)
    
    task = asyncio.create_task(client.connect())
    
    for _ in range(20):
        if mock_server.ws: break
        await asyncio.sleep(0.1)
    
    await mock_server.ws.send_json({"event": "test"})
    
    for _ in range(20):
        if len(results) > 0: break
        await asyncio.sleep(0.1)
        
    assert len(results) == 1
    assert results[0]["event"] == "test"
    
    await client.stop()
    task.cancel()

@pytest.mark.asyncio
async def test_ws_auto_reconnect(mock_server):
    url = f"ws://127.0.0.1:{mock_server.port}/ws/market"
    client = ClobWsClient(url=url, asset_ids=["token_reconnect"])
    client._reconnect_delay = 0.1
    task = asyncio.create_task(client.connect())
    
    for _ in range(20):
        if mock_server.ws: break
        await asyncio.sleep(0.1)
    assert mock_server.ws is not None
    
    await mock_server.stop()
    await asyncio.sleep(0.5)
    
    mock_server.messages_received = []
    await mock_server.start()
    # Update the client's URL to the new port
    client.url = f"ws://127.0.0.1:{mock_server.port}/ws/market"
    
    success = False
    for _ in range(100):
        if any(m.get("operation") == "subscribe" for m in mock_server.messages_received):
            success = True
            break
        await asyncio.sleep(0.1)
    assert success
    
    await client.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

@pytest.mark.asyncio
async def test_ws_malformed_json(mock_server):
    url = f"ws://127.0.0.1:{mock_server.port}/ws/market"
    client = ClobWsClient(url=url)
    
    results = []
    client.register_callback(lambda d: results.append(d))
    
    task = asyncio.create_task(client.connect())
    
    for _ in range(20):
        if mock_server.ws: break
        await asyncio.sleep(0.1)
        
    # Send malformed text
    await mock_server.ws.send_str("not a json")
    # Send valid json after
    await mock_server.ws.send_json({"status": "ok"})
    
    for _ in range(20):
        if len(results) > 0: break
        await asyncio.sleep(0.1)
        
    assert len(results) == 1
    assert results[0]["status"] == "ok"
    
    await client.stop()
    task.cancel()
