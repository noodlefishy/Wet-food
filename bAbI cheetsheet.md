# bAbI Tasks for Pico

| # | Task | Example | Fit |
|---|---|---|---|
| 1 | Single Supporting Fact | John is in the playground. **Where is John?** → playground | Yes |
| 2 | Two Supporting Facts | John is in the kitchen; he picks up milk, then goes to the garden. **Where is the milk?** → garden | Yes |
| 3 | Three Supporting Facts | John picks up a ball, goes to the bedroom, drops it, then goes to the garden. **Where is the ball?** → bedroom | Hard |
| 4 | Two Argument Relations | Office is north of kitchen. **What is north of the kitchen?** → office | Yes |
| 5 | Three Argument Relations | Bill gives milk to John; John gives it to Mary. **Who got the milk from John?** → Mary | Yes |
| 6 | Yes/No Questions | John is in the playground. **Is John in the playground?** → yes | Yes |
| 7 | Counting | Daniel picks up a ball and a fish. **How many objects is he holding?** → two | Yes, small counts |
| 8 | Lists / Sets | Daniel picks up a ball and a fish. **What is he holding?** → ball, fish | Partial |
| 9 | Simple Negation | Sandra is not in the garden. **Is Sandra in the garden?** → no | Yes |
| 10 | Indefinite Knowledge | John is in the classroom or playground, not the classroom. **Where is John?** → playground | Yes |
| 11 | Basic Coreference | Daniel is in the kitchen; then he goes to the garden. **Where is Daniel?** → garden | Yes |
| 12 | Conjunction | Mary and John go to the kitchen. **Where is Mary?** → kitchen | Yes |
| 13 | Compound Coreference | Daniel and Sandra go to the garden; then they go to the office. **Where is Daniel?** → office | Yes |
| 14 | Time Reasoning | John is first in the park, then at school. **Where was he before school?** → park | Yes |
| 15 | Basic Deduction | Mice fear cats; Jessica is a mouse. **What is Jessica afraid of?** → cats | Yes |
| 16 | Basic Induction | Lily is a white swan; Greg is a swan. **What colour is Greg?** → white | Yes |
| 17 | Positional Reasoning | Ball is left of box; fish is left of ball. **Is fish left of box?** → yes | Yes |
| 18 | Size Reasoning | Box is bigger than ball; ball is bigger than fish. **What is smaller than the box?** → ball | Yes |
| 19 | Path Finding | Kitchen is north of hallway; bathroom is east of kitchen. **How do you get to the bathroom?** → north, east | No |
| 20 | Agent's Motivations | John is hungry and goes to the kitchen. **Why?** → hungry | Yes |

## Suggested priorities

- **Already supported:** 1, 2, 5, 6, 9, 10, 14, 15, 16, 20
- **Good next additions:** 7 (small counts), 11 and 13 (pronouns), 18 (size comparisons)
- **More challenging:** 3 (longer fact chains), 8 (lists), 19 (route planning)

Use task names in `config/logic.json` (for example, `bAbI Task 01: Single Location`) to label benchmark results.
