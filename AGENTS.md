# Doc management
- Always update the README to align with the latest changes

# Design guideline
1. **Prime Directive: never design against the requirements**
   Design the architecture to *absorb change*, not to mirror today’s use cases. ([InfoQ][1])

2. **Decompose by volatility, not by function/domain**
   Identify likely areas of change and make them your building blocks; avoid functional/domain decomposition. ([InfoQ][1], [idesign.net][2])

3. **Encapsulate volatility (“vaults”)**
   Each component/service contains a specific kind of change so requirement changes don’t ripple across the system. ([InfoQ][1], [idesign.net][2])

4. **Compose behavior from stable building blocks**
   Satisfy use cases by integrating services; the architecture stays static as use cases vary (“Composable Design,” “There Is No Feature”). ([PagePlace][3])

5. **Use the standard roles to classify components**
   Structure around Clients, Managers, Engines, Resource Access, Resources (+ Utilities) to reflect common volatilities. ([PagePlace][3], [idesign.net][4])

6. **Keep business logic out of clients**
   Clients are the most volatile; push rules and workflows into Managers/Engines. ([InfoQ][1], [Bookey][5])

7. **Managers orchestrate, Engines decide**
   Managers handle workflows/sequencing; Engines implement rules/algorithms. Mind the Managers-to-Engines ratio. ([PagePlace][3], [Bookey][5])

8. **Isolate resource access as its own volatility**
   Separate *how* you access resources from *what* the resources are; both can change. ([PagePlace][3], [se-radio.net][6])

9. **Prefer closed (or semi-closed) architectures**
   Constrain who can call whom to control coupling and keep independence/stability. ([PagePlace][3], [Bookey][5])

10. **Layer consistently**
    Typical layers: Client → Business Logic (Managers/Engines) → ResourceAccess → Resource, with a Utilities bar. ([PagePlace][3])

11. **Classify responsibilities with the “Four Questions”**
    Use Löwy’s classification questions to decide the right role for a capability. ([PagePlace][3])

12. **Smallest useful set of services (\~order of 10)**
    Aim for the minimal number of composable services that cover the *core* use cases. ([InfoQ][1])

13. **Distinguish volatility vs. variability; map axes of volatility**
    Hunt for true *volatility* (business change) vs. code-level variability; use explicit axes to find it. ([PagePlace][3], [InformIT][7])

14. **Keep the architecture static across versions**
    Contain change inside components; the high-level structure shouldn’t churn. ([PagePlace][3])

15. **Resist functional/domain decomposition**
    It creates either tiny fragments with high integration cost or blobs with high internal complexity and pollutes clients. ([InfoQ][1])

16. **Design for communication and onboarding**
    Using common roles/semantics accelerates comprehension and keeps teams aligned. ([idesign.net][2], [PagePlace][3])

17. **Validate the system design early**
    Separate architecture (fast, upfront) from detailed design (during development); avoid “big upfront *detailed* design.” ([idesign.net][2], [PagePlace][3])

18. **Be microservice-agnostic**
    Service granularity follows volatility and composition, not a “microservices first” rule. ([PagePlace][3])

19. **Name and structure components explicitly**
    Clear names and semantics (“What’s in a Name”) enforce intent and make designs reviewable. ([PagePlace][3])

20. **Strive for symmetry**
    Keep analogous parts shaped the same; symmetrical structures reduce accidental complexity. ([PagePlace][3])

If you want, I can map these to your LangGraph agent platform with concrete examples of “Managers vs. Engines vs. ResourceAccess” nodes and how to keep clients thin.

[1]: https://www.infoq.com/articles/book-review-righting-software/ "Q&A on the Book Righting Software - InfoQ"
[2]: https://idesign.net/assets/documents/IDesign-Method-Management-Overview.pdf "Microsoft Word - IDesign-Method-Management-Overview.docx"
[3]: https://api.pageplace.de/preview/DT0400.9780136523987_A41316644/preview-9780136523987_A41316644.pdf "Righting Software"
[4]: https://www.idesign.net/Training/Architect-Master-Class?utm_source=chatgpt.com "Architect's Master Class"
[5]: https://cdn.bookey.app/files/pdf/book/en/righting-software.pdf?utm_source=chatgpt.com "Righting Software PDF"
[6]: https://se-radio.net/2020/04/episode-407-juval-lowy-on-righting-software/?utm_source=chatgpt.com "SE Radio 407: Juval Löwy on Righting Software"
[7]: https://www.informit.com/articles/article.aspx?p=2995357&seqNum=3&utm_source=chatgpt.com "Identifying Volatility | Software System Decomposition - InformIT"


# Coding guidelines
Always follow these guidelines when writing code:
1. Use intention-revealing names
   — Pronounceable, searchable, no encodings; name by purpose.
2. Keep functions small
   — Few lines, minimal branches.
3. Functions do one thing
   — No mixed responsibilities; extract until each has a single concern.
4. Minimize parameters
   — Prefer 0–2 args; avoid boolean flags and long parameter lists.
5. Separate command from query (CQS)
   — A routine either changes state or returns information, not both.
6. Avoid side effects; prefer immutability
   — Eliminate hidden temporal coupling; limit shared mutable state.
7. Make the code communicate
   — Self-documenting structure; comments only for “why,” not “what/how.”
8. Format for readability
   — Consistent layout, sensible spacing/indentation, meaningful vertical ordering.
9. Eliminate duplication (DRY)
   — Factor out repeats; duplication breeds divergence.
10. Prefer polymorphism/composition over conditionals
    — Replace big switch/if chains with strategy, state, or polymorphic dispatch.
11. Law of Demeter (“don’t talk to strangers”)
    — Avoid train-wreck call chains; interact with immediate collaborators.
12. Tell, don’t ask
    — Expose behaviors, not data; choose objects vs. data structures deliberately.
13. Handle errors with exceptions, not return codes
    — Narrow try/catch blocks; add context; avoid swallowed errors.
14. Isolate external boundaries
    — Wrap third-party APIs; keep dependencies at the edges.
15. Single Responsibility Principle (SRP)
    — One reason to change per module/class.
16. Open–Closed Principle (OCP)
    — Open for extension, closed for modification.
17. Liskov Substitution Principle (LSP)
    — Subtypes must be substitutable for their base types without surprises.
18. Interface Segregation Principle (ISP)
    — Many small, focused interfaces over one “fat” interface.
19. Dependency Inversion Principle (DIP)
    — Depend on abstractions, not concretions; invert source-code dependencies.
20. Simple, emergent design + keep it clean
    — Follows Kent Beck’s four rules (passes tests, no duplication, expresses intent, minimal elements) and the Boy Scout Rule: leave the code cleaner than you found it.

