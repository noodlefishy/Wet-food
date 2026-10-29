import time, json, statistics, urllib.parse, urllib.request

MEOW_HOST = "http://192.168.4.1"

TEST_PROMPTS = [
    "what are your names ?",
    "who is meow ?",
    "who is miaou ?",
    "what do cats drink ?",
    "once upon a time",
    "the little cat",
]

def runBenchmarkQuery(prompt):
    url = f"{MEOW_HOST}/gen?p={urllib.parse.quote(prompt)}"
    req = urllib.request.Request(url)
    tStart = time.perf_counter()
    firstTokenTime = None
    latencies, tokens, ramSnapshots = [], [], []

    with urllib.request.urlopen(req, timeout=120) as response:
        buffer = ""
        while True:
            chunk = response.read(64)
            if not chunk: break
            buffer += chunk.decode('utf-8', errors='ignore')
            lines = buffer.split('\n')
            buffer = lines.pop()
            for line in lines:
                line = line.strip()
                if not line or not line.startswith('{'): continue
                try:
                    data = json.loads(line)
                    now = time.perf_counter()
                    if firstTokenTime is None: firstTokenTime = now - tStart
                    tokens.append(data.get("w", ""))
                    latencies.append(data.get("ms", 0))
                    ramSnapshots.append(data.get("ram", 0))
                except json.JSONDecodeError: continue

    return {
        "prompt": prompt,
        "tokens": tokens,
        "tokenCount": len(tokens),
        "totalTime": time.perf_counter() - tStart,
        "ttftSec": firstTokenTime or 0.0,
        "latenciesMs": latencies,
        "ramSnapshots": ramSnapshots,
    }

def printReport(results):
    print("=" * 65)
    print("      MEOW & MIAOU DISTRIBUTED NEURAL ENGINE BENCHMARK      ")
    print("=" * 65)
    allLatencies, allTps = [], []
    for r in results:
        tCount = r["tokenCount"]
        if tCount == 0:
            print(f"FAILED: '{r['prompt']}' (No tokens received)")
            continue
        tpot = statistics.mean(r["latenciesMs"]) if r["latenciesMs"] else 0
        tps = (tCount / r["totalTime"])
        allLatencies.extend(r["latenciesMs"])
        allTps.append(tps)
        outputPreview = " ".join(r["tokens"])
        print(f"\nPrompt:  \"{r['prompt']}\"")
        print(f"Output:  \"{outputPreview[:60]}...\"")
        print(f"Tokens:  {tCount} tokens | TTFT: {r['ttftSec']:.2f}s | Avg TPOT: {tpot:.1f}ms | TPS: {tps:.2f} tok/s")
        if r["ramSnapshots"]:
            print(f"Meow RAM (min / max): {min(r['ramSnapshots'])} / {max(r['ramSnapshots'])} bytes free")

    if allLatencies:
        print("\n" + "-" * 65)
        print("AGGREGATE PERFORMANCE SUMMARY")
        print("-" * 65)
        sortedLat = sorted(allLatencies)
        print(f"Average Throughput : {statistics.mean(allTps):.2f} tokens/sec")
        print(f"Mean Token Latency : {statistics.mean(allLatencies):.1f} ms")
        print(f"p50 Token Latency  : {sortedLat[int(len(sortedLat) * 0.50)]} ms")
        print(f"p95 Token Latency  : {sortedLat[int(len(sortedLat) * 0.95)]} ms")
        print("=" * 65)

if __name__ == "__main__":
    print(f"Starting test suite against {MEOW_HOST}...")
    benchmarkData = []
    for p in TEST_PROMPTS:
        print(f"Running benchmark on: '{p}'")
        try: benchmarkData.append(runBenchmarkQuery(p))
        except Exception as e: print(f"Error querying Meow: {e}")
        time.sleep(1)
    printReport(benchmarkData)
