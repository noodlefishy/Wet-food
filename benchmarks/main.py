import time
import json
import statistics
import urllib.parse
import urllib.request

MEOW_HOST = "http://192.168.4.1"

TEST_PROMPTS = [
    "what are your names ?",
    "who is meow ?",
    "who is miaou ?",
    "what do cats drink ?",
    "once upon a time",
    "the little cat",
]

def run_benchmark_query(prompt):
    url = f"{MEOW_HOST}/gen?p={urllib.parse.quote(prompt)}"
    req = urllib.request.Request(url)
    
    t_start = time.perf_counter()
    first_token_time = None
    latencies = []
    tokens = []
    ram_snapshots = []
    
    with urllib.request.urlopen(req, timeout=120) as response:
        buffer = ""
        while True:
            chunk = response.read(64)
            if not chunk:
                break
            buffer += chunk.decode('utf-8', errors='ignore')
            lines = buffer.split('\n')
            buffer = lines.pop()
            
            for line in lines:
                line = line.strip()
                if not line or not line.startswith('{'):
                    continue
                try:
                    data = json.loads(line)
                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now - t_start
                    
                    tokens.append(data.get("w", ""))
                    latencies.append(data.get("ms", 0))
                    ram_snapshots.append(data.get("ram", 0))
                except json.JSONDecodeError:
                    continue

    t_total = time.perf_counter() - t_start
    return {
        "prompt": prompt,
        "tokens": tokens,
        "token_count": len(tokens),
        "total_time": t_total,
        "ttft_sec": first_token_time or 0.0,
        "latencies_ms": latencies,
        "ram_snapshots": ram_snapshots,
    }

def print_report(results):
    print("=" * 65)
    print("      MEOW & MIAOU DISTRIBUTED NEURAL ENGINE BENCHMARK      ")
    print("=" * 65)
    
    all_latencies = []
    all_tps = []
    
    for r in results:
        t_count = r["token_count"]
        if t_count == 0:
            print(f"FAILED: '{r['prompt']}' (No tokens received)")
            continue
            
        tpot = statistics.mean(r["latencies_ms"]) if r["latencies_ms"] else 0
        tps = (t_count / r["total_time"])
        all_latencies.extend(r["latencies_ms"])
        all_tps.append(tps)
        
        output_preview = " ".join(r["tokens"])
        print(f"\nPrompt:  \"{r['prompt']}\"")
        print(f"Output:  \"{output_preview[:60]}...\"")
        print(f"Tokens:  {t_count} tokens | TTFT: {r['ttft_sec']:.2f}s | Avg TPOT: {tpot:.1f}ms | TPS: {tps:.2f} tok/s")
        if r["ram_snapshots"]:
            print(f"Meow RAM (min / max): {min(r['ram_snapshots'])} / {max(r['ram_snapshots'])} bytes free")

    if all_latencies:
        print("\n" + "-" * 65)
        print("AGGREGATE PERFORMANCE SUMMARY")
        print("-" * 65)
        sorted_lat = sorted(all_latencies)
        p50 = sorted_lat[int(len(sorted_lat) * 0.50)]
        p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
        
        print(f"Average Throughput : {statistics.mean(all_tps):.2f} tokens/sec")
        print(f"Mean Token Latency : {statistics.mean(all_latencies):.1f} ms")
        print(f"p50 Token Latency  : {p50} ms")
        print(f"p95 Token Latency  : {p95} ms")
        print("=" * 65)

if __name__ == "__main__":
    print(f"Starting test suite against {MEOW_HOST}...")
    benchmark_data = []
    for p in TEST_PROMPTS:
        print(f"Running benchmark on: '{p}'")
        try:
            res = run_benchmark_query(p)
            benchmark_data.append(res)
        except Exception as e:
            print(f"Error querying Meow: {e}")
        time.sleep(1)
        
    print_report(benchmark_data)

"""
=================================================================
      MEOW & MIAOU DISTRIBUTED NEURAL ENGINE BENCHMARK      
=================================================================

Prompt:  "what are your names ?"
Output:  "we are meow and miaou ...."
Tokens:  6 tokens | TTFT: 38.84s | Avg TPOT: 4270.5ms | TPS: 0.11 tok/s
Meow RAM (min / max): 4176 / 59264 bytes free

Prompt:  "who is meow ?"
Output:  "meow is the pico wh who talks to you on the [Miaou disconnec..."
Tokens:  12 tokens | TTFT: 34.98s | Avg TPOT: 4063.4ms | TPS: 0.15 tok/s
Meow RAM (min / max): 0 / 61888 bytes free

Prompt:  "who is miaou ?"
Output:  "[Miaou sync error]..."
Tokens:  1 tokens | TTFT: 0.09s | Avg TPOT: 0.0ms | TPS: 10.58 tok/s
Meow RAM (min / max): 0 / 0 bytes free

Prompt:  "what do cats drink ?"
Output:  "[Miaou sync error]..."
Tokens:  1 tokens | TTFT: 0.26s | Avg TPOT: 0.0ms | TPS: 3.81 tok/s
Meow RAM (min / max): 0 / 0 bytes free

Prompt:  "once upon a time"
Output:  "[Miaou sync error]..."
Tokens:  1 tokens | TTFT: 0.41s | Avg TPOT: 0.0ms | TPS: 2.41 tok/s
Meow RAM (min / max): 0 / 0 bytes free

Prompt:  "the little cat"
Output:  "had an idea ...."
Tokens:  4 tokens | TTFT: 15.03s | Avg TPOT: 3776.8ms | TPS: 0.18 tok/s
Meow RAM (min / max): 20864 / 81168 bytes free

-----------------------------------------------------------------
AGGREGATE PERFORMANCE SUMMARY
-----------------------------------------------------------------
Average Throughput : 2.87 tokens/sec
Mean Token Latency : 3579.6 ms
p50 Token Latency  : 4133 ms
p95 Token Latency  : 4877 ms
=================================================================

"""