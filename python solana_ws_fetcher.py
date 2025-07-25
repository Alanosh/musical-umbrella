import json
import os
import asyncio
import websockets
import pandas as pd
import numpy as np
from solders.pubkey import Pubkey
from solana.rpc.async_api import AsyncClient
from solana.rpc.websocket_api import connect
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any

# Constants
SOLANA_WS_URL = "wss://api.mainnet-beta.solana.com"
DATA_DIR = "memecoin_data"
OUTPUT_JSON = "realtime_data.json"
ERROR_LOG = "fetcher_errors.log"
INTERVAL_SECONDS = 10
INTERVALS_PER_COIN = 361
LAST_VALID_JSON = "last_valid_data.json"
FEATURES = [
    'Price_Surge', 'Volume_SOL', 'Insider_Ratio', 'Wallet_Diversity', 
    'Liquidity_Volume', 'Price_Volatility', 'Price_Momentum', 
    'Transaction_Frequency', 'Whale_Volume', 'Trade_Size_Variance', 
    'Sniper_Bot_Activity', 'Buy_Sell_Ratio', 'Failed_Trade_Ratio', 'TOTAL_NET_SOL'
]

# Logging Setup
logging.basicConfig(
    filename=ERROR_LOG,
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def fetch_memecoin_addresses(client: AsyncClient) -> List[str]:
    """Fetch 10 memecoin addresses from the public Solana endpoint."""
    try:
        # Query recent transactions to identify memecoin addresses
        signatures = await client.get_signatures_for_address(
            Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"),  # SPL Token Program
            limit=100
        )
        addresses = set()
        for sig in signatures.value:
            tx = await client.get_transaction(sig.signature)
            if tx.value and tx.value.transaction:
                meta = tx.value.transaction.meta
                if meta and meta.inner_instructions:
                    for instr in meta.inner_instructions:
                        for inner in instr.instructions:
                            if inner.program_id == Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"):
                                for key in inner.accounts:
                                    addr = str(key)
                                    # Heuristic: Memecoins with high recent activity and newly created
                                    tx_count = sum(1 for s in signatures.value if any(str(k) == addr for k in s.accounts))
                                    if tx_count > 25 and tx.value.block_time > (int(datetime.utcnow().timestamp()) - 86400):  # Created within last 24 hours
                                        addresses.add(addr)
                                        if len(addresses) >= 10:
                                            return list(addresses)[:10]
        return list(addresses)[:10] if addresses else []
    except Exception as e:
        logging.error(f"Fetch memecoin addresses error: {str(e)}")
        return []

# Initialize Data Cache
data_cache: Dict[str, List[Dict[str, Any]]] = {}
last_valid_data: Dict[str, List[Dict[str, Any]]] = {}

async def fetch_transaction_data(client: AsyncClient, address: str) -> Dict[str, Any]:
    """Fetch transaction data for a memecoin address."""
    try:
        pubkey = Pubkey.from_string(address)
        signatures = await client.get_signatures_for_address(pubkey, limit=100)
        if not signatures.value:
            raise ValueError(f"No recent transactions for {address}")
        
        tx_data = []
        for sig in signatures.value:
            tx = await client.get_transaction(sig.signature)
            if tx.value and tx.value.transaction:
                meta = tx.value.transaction.meta
                if meta and meta.inner_instructions:
                    price = meta.post_balances[0] / meta.pre_balances[0] - 1 if meta.pre_balances[0] else 0
                    volume = meta.post_balances[0] / 1e9  # Convert lamports to SOL
                    tx_data.append({
                        'timestamp': tx.value.block_time,
                        'price': price,
                        'volume': volume,
                        'accounts': len(set(meta.account_keys))
                    })
        return {'address': address, 'transactions': tx_data}
    except Exception as e:
        logging.error(f"Fetch error for {address}: {str(e)}")
        return {'address': address, 'transactions': []}

def aggregate_interval_data(tx_data: Dict[str, Any], interval_start: datetime) -> Dict[str, Any]:
    """Aggregate transaction data into 10-second interval."""
    try:
        address = tx_data['address']
        transactions = [t for t in tx_data['transactions'] 
                       if interval_start <= datetime.fromtimestamp(t['timestamp']) < interval_start + timedelta(seconds=INTERVAL_SECONDS)]
        if not transactions:
            return None
        
        prices = [t['price'] for t in transactions]
        volumes = [t['volume'] for t in transactions]
        accounts = set()
        for t in transactions:
            accounts.update(t['accounts'])
        
        # Compute Features
        data = {
            'Contract_Address': address,
            'Interval': (interval_start - datetime(1970, 1, 1)).total_seconds() // INTERVAL_SECONDS,
            'Price_Surge': np.mean(prices) if prices else 0,
            'Volume_SOL': sum(volumes),
            'Insider_Ratio': len([v for v in volumes if v > np.mean(volumes) + 2 * np.std(volumes)]) / max(len(volumes), 1),
            'Wallet_Diversity': len(accounts) / max(len(transactions), 1),
            'Liquidity_Volume': sum(volumes) * 0.8,  # Simplified proxy
            'Price_Volatility': np.std(prices) if len(prices) > 1 else 0,
            'Price_Momentum': (prices[-1] - prices[0]) / prices[0] if prices and prices[0] else 0,
            'Transaction_Frequency': len(transactions) / INTERVAL_SECONDS,
            'Whale_Volume': sum([v for v in volumes if v > np.mean(volumes) + 2 * np.std(volumes)]),
            'Trade_Size_Variance': np.var(volumes) if volumes else 0,
            'Sniper_Bot_Activity': sum(1 for t in transactions if t['timestamp'] < interval_start.timestamp() + 2) / max(len(transactions), 1),
            'Buy_Sell_Ratio': sum(1 for p in prices if p > 0) / max(sum(1 for p in prices if p < 0), 1),
            'Failed_Trade_Ratio': sum(1 for t in transactions if t.get('status') == 'failed') / max(len(transactions), 1),
            'TOTAL_NET_SOL': sum(volumes)
        }
        
        # Pre-Filtering (Inspired by @SolanaSniperX, @RugPullRadar)
        if data['Sniper_Bot_Activity'] > 0.8 and data['Interval'] < 90:
            data['Priority'] = 'High'
        elif data['Liquidity_Volume'] < 0.4 * np.mean(volumes) and data['Insider_Ratio'] > 0.75:
            data['Priority'] = 'Low'  # Potential honeypot
        else:
            data['Priority'] = 'Medium'
        
        return data
    except Exception as e:
        logging.error(f"Aggregation error for {tx_data['address']}: {str(e)}")
        return None

async def save_interval_data():
    """Save aggregated data to JSON and cache."""
    try:
        interval_start = datetime.utcnow().replace(second=0, microsecond=0) - timedelta(seconds=INTERVAL_SECONDS)
        output_data = []
        
        async with AsyncClient(SOLANA_WS_URL) as client:
            # Fetch memecoin addresses if cache is empty
            if not data_cache:
                memecoin_addresses = await fetch_memecoin_addresses(client)
                for addr in memecoin_addresses:
                    data_cache[addr] = []
                    last_valid_data[addr] = []
            
            for address in data_cache.keys():
                tx_data = await fetch_transaction_data(client, address)
                interval_data = aggregate_interval_data(tx_data, interval_start)
                if interval_data:
                    data_cache[address].append(interval_data)
                    if len(data_cache[address]) > INTERVALS_PER_COIN:
                        data_cache[address] = data_cache[address][-INTERVALS_PER_COIN:]
                    output_data.append(interval_data)
                    last_valid_data[address] = data_cache[address].copy()
        
        if output_data:
            with open(OUTPUT_JSON, 'w') as f:
                json.dump(output_data, f)
            with open(LAST_VALID_JSON, 'w') as f:
                json.dump(last_valid_data, f)
    except Exception as e:
        logging.error(f"Save error: {str(e)}")

async def websocket_loop():
    """Main WebSocket loop with reconnection logic."""
    retry_delay = 1
    while True:
        try:
            async with connect(SOLANA_WS_URL) as ws:
                for address in data_cache.keys():
                    await ws.account_subscribe(Pubkey.from_string(address))
                while True:
                    await save_interval_data()
                    await asyncio.sleep(INTERVAL_SECONDS)
        except (websockets.exceptions.ConnectionClosed, Exception) as e:
            logging.error(f"WebSocket error: {str(e)}")
            retry_delay = min(retry_delay * 2, 16)  # Exponential backoff
            await asyncio.sleep(retry_delay)

def main():
    """Run the WebSocket fetcher."""
    try:
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        asyncio.run(websocket_loop())
    except Exception as e:
        logging.error(f"Main error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
