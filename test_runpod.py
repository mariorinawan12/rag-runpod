import asyncio
import os
import aiohttp
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")

if not API_KEY or not ENDPOINT_ID:
    print("❌ ERROR: RUNPOD_API_KEY or RUNPOD_ENDPOINT_ID not set in .env")
    exit(1)

BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

async def run_test():
    print(f"🚀 Testing RunPod Endpoint: {ENDPOINT_ID}")
    
    async with aiohttp.ClientSession() as session:
        # 1. Test Embed
        print("\nDATA 1: Testing Embedding...")
        payload_embed = {
            "input": {
                "action": "embed",
                "text": "Hello RunPod!"
            }
        }
        
        try:
            async with session.post(f"{BASE_URL}/runsync", json=payload_embed, headers=HEADERS) as resp:
                result = await resp.json()
                if "output" in result:
                    print("✅ Embedding Success! Vector length:", len(result["output"]["embedding"]))
                else:
                    print("❌ Embedding Failed:", result)
        except Exception as e:
            print(f"❌ Embedding Error: {e}")

        # 2. Test LLM Generation
        print("\nData 2: Testing LLM Generation...")
        payload_llm = {
            "input": {
                "action": "generate",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "max_tokens": 50
            }
        }

        try:
            # Note: Using runsync for quick test, usually use run + status polling
            async with session.post(f"{BASE_URL}/runsync", json=payload_llm, headers=HEADERS) as resp:
                result = await resp.json()
                if "output" in result:
                    print("✅ LLM Success!")
                    print("Response:", result["output"]["text"])
                else:
                    print("❌ LLM Failed:", result)
        except Exception as e:
            print(f"❌ LLM Error: {e}")

if __name__ == "__main__":
    asyncio.run(run_test())
