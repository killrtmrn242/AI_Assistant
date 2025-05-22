from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os
import httpx
import re
import asyncio  # Added this import
from difflib import get_close_matches

# Coin mapping for better symbol resolution
COIN_MAPPING = {
    "btc": "bitcoin",
    "bitcoin": "bitcoin",
    "eth": "ethereum",
    "ethereum": "ethereum",
    "sol": "solana",
    "solana": "solana",
    "ada": "cardano",
    "cardano": "cardano",
    "doge": "dogecoin",
    "dogecoin": "dogecoin",
    "xrp": "ripple",
    "ripple": "ripple",
    "dot": "polkadot",
    "polkadot": "polkadot",
}

load_dotenv()  # Load environment variables from .env

app = FastAPI()

# Serve static files (CSS, JS, images) from /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the HTML file from the root
@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

# Load environment variables
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")
CRYPTO_PANIC_API_KEY = os.getenv("CRYPTO_PANIC_API_KEY")
COINGECKO_API_URL = os.getenv("COINGECKO_API_URL")
COINMARKETCAP_API_KEY = os.getenv("COINMARKETCAP_API_KEY")

# Validate required environment variables
if not OLLAMA_API_URL:
    raise ValueError("OLLAMA_API_URL is not set in environment variables.")
if not CRYPTO_PANIC_API_KEY:
    raise ValueError("CRYPTO_PANIC_API_KEY is not set in environment variables.")
if not COINGECKO_API_URL:
    raise ValueError("COINGECKO_API_URL is not set in environment variables.")

def clean_token(token: str) -> str:
    """Clean, normalize and approximate match a token/symbol input"""
    # Remove all non-alphabetic characters
    cleaned = re.sub(r'[^a-zA-Z]', '', token).lower()

    if cleaned in COIN_MAPPING:
        return COIN_MAPPING[cleaned]

    # Попробуем найти наиболее похожее название среди ключей COIN_MAPPING
    close_matches = get_close_matches(cleaned, COIN_MAPPING.keys(), n=1, cutoff=0.6)
    if close_matches:
        return COIN_MAPPING[close_matches[0]]

    return cleaned  # Вернём очищенный, если не удалось найти совпадение


async def get_crypto_news(token: str):
    """Fetch news from CryptoPanic for a specific token"""
    token_cleaned = clean_token(token)
    if not token_cleaned:
        return []
    
    url = "https://cryptopanic.com/api/v1/posts/"
    params = {
        "auth_token": CRYPTO_PANIC_API_KEY,
        "currencies": token_cleaned.upper(),
        "kind": "news"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            news = []
            for item in data.get("results", [])[:3]:
                news.append({
                    "title": item.get("title", "No title"),
                    "url": item.get("url", "#"),
                    "source": item.get("source", {}).get("title", "Unknown")
                })
            return news
        except Exception as e:
            print(f"CryptoPanic API error: {str(e)}")
            return []


async def get_price_and_marketcap_coingecko(token: str):
    token_cleaned = clean_token(token)
    if not token_cleaned:
        return None

    url = f"{COINGECKO_API_URL}/coins/markets"
    params = {
        "vs_currency": "usd",
        "ids": token_cleaned,
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data:
                coin = data[0]
                return {
                    "price": coin.get("current_price"),
                    "market_cap": coin.get("market_cap"),
                    "rank": coin.get("market_cap_rank")
                }
            return None
        except httpx.HTTPStatusError as e:
            print(f"CoinGecko API error: {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error with CoinGecko API: {str(e)}")
            return None

async def get_price_and_marketcap_coinmarketcap(token: str):
    if not COINMARKETCAP_API_KEY:
        return None

    token_cleaned = clean_token(token)
    if not token_cleaned:
        return None

    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
    headers = {"X-CMC_PRO_API_KEY": COINMARKETCAP_API_KEY}
    params = {"symbol": token_cleaned.upper()}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "data" in data and token_cleaned.upper() in data["data"]:
                coin = data["data"][token_cleaned.upper()]
                quote = coin["quote"]["USD"]
                return {
                    "price": quote.get("price"),
                    "market_cap": quote.get("market_cap"),
                    "rank": coin.get("cmc_rank")
                }
            return None
        except httpx.HTTPStatusError as e:
            print(f"CoinMarketCap API error: {e.response.text if hasattr(e, 'response') else str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error with CoinMarketCap API: {str(e)}")
            return None


async def ask_ollama(prompt: str):
    """Query Ollama LLM with the given prompt"""
    url = f"{OLLAMA_API_URL}/api/generate"
    json_data = {
        "model": "llama2",  # Change to your preferred model
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9
        }
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=json_data, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "No response from Ollama")
        except httpx.HTTPStatusError as e:
            print(f"Ollama API error: {e.response.text if hasattr(e, 'response') else str(e)}")
            return f"Error querying Ollama: {str(e)}"
        except Exception as e:
            print(f"Unexpected error with Ollama API: {str(e)}")
            return f"Error querying Ollama: {str(e)}"

@app.post("/query")
async def handle_query(request: Request):
    data = await request.json()
    user_query = data.get("query", "").strip()
    
    if not user_query:
        return JSONResponse({"answer": "Please enter a query."})

    # Extract and clean token
    last_word = user_query.split()[-1]
    token = clean_token(last_word)
    
    if not token:
        return JSONResponse({
            "answer": "Could not identify cryptocurrency in your query. Please specify a valid cryptocurrency name or symbol."
        })

    try:
        # Fetch data concurrently
        news_task = get_crypto_news(token)
        coingecko_task = get_price_and_marketcap_coingecko(token)
        news, price_data = await asyncio.gather(news_task, coingecko_task)
        
        # Fallback to CoinMarketCap if CoinGecko fails and API key exists
        if not price_data and COINMARKETCAP_API_KEY:
            price_data = await get_price_and_marketcap_coinmarketcap(token)

        if not price_data:
            return JSONResponse({
                "answer": f"Could not find data for {token.upper()}. Please check the cryptocurrency name and try again."
            })

        # Format news for display
        formatted_news = []
        for item in news:
            formatted_news.append(f"{item['title']} (Source: {item['source']}, {item['url']})")

        # Prepare context for LLM
        context = f"""
User asked: "{user_query}"

Information about {token.upper()}:

Current Price: ${price_data['price']:,.2f}
Market Cap: ${price_data['market_cap']:,.2f}
Market Rank: #{price_data['rank']}

Recent News:
{chr(10).join(formatted_news) if formatted_news else "No recent news found"}
"""

        # Get response from Ollama
        response = await ask_ollama(f"{context}\n\nPlease provide a helpful response to the user's question:")

        return JSONResponse({
            "answer": response,
            "data": {
                "price": price_data['price'],
                "market_cap": price_data['market_cap'],
                "rank": price_data['rank'],
                "news": news
            }
        })
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return JSONResponse({
            "answer": "An error occurred while processing your request. Please try again later.",
            "error": str(e)
        }, status_code=500)