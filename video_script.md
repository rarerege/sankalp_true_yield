# TrueYield video script (4 minutes)

About 575 words, which is about 4 minutes at a calm pace. Record in the four parts marked below and join them.

Order: slides 1 to 3, then the live app, then the PDF report, then slides 6 to 8.

## Part A: slides (0:00 to 1:10)

**Slide 1 (0:00 to 0:15)**

I'm Raghav Rege, a third-year computer science student at SRM Institute of Science and Technology. This is TrueYield: an independent check on how much energy a rooftop solar system is losing, and how much a fix wins back.

**Slide 2 (0:15 to 0:40)**

A small business puts solar on its roof, often on a loan. Then dust, shade and faults build up, and the savings shrink without anyone noticing. Field tests on Delhi rooftops found ten to twenty percent more energy after a single cleaning. The panels still work, so nobody sees the loss. And no neutral record says what the system should be producing.

**Slide 3 (0:40 to 1:10)**

TrueYield works as one loop. It takes the inverter's own data and free satellite weather, works out what the system should have produced, and measures the gap, always with a range. This chart is real data from a rooftop plant in Jaipur in 2022. Navy is expected output, amber is actual. The gap in December is a twenty-seven day breakdown. The gap in May and June closes after heavy rain on the nineteenth of June.

## Part B: live demo (1:10 to 2:25)

**Terminal, showing the finished run (1:10 to 1:32)**

Here is the working prototype. One command runs the whole pipeline. The input is one year of daily energy readings from a 120 kilowatt rooftop plant in Jaipur, a public dataset, together with satellite weather from NASA. Nothing is trained: there is no machine learning here. Each system is compared against its own best performance.

**App, Jaipur selected, top of the page (1:22 to 1:50)**

In the viewer I choose the Jaipur system. Every screen carries a data label, so it is clear this is not my own system. The estimated loss is 8.4 percent, with a range of 6.5 to 9.2, about fourteen thousand nine hundred kilowatt-hours. The tool never shows a loss without its range. It also states its detection limit: on this data, anything under about nine percent over two weeks cannot be told apart from noise, so it is not claimed.

**App, scroll to the second chart (1:50 to 2:05)**

This chart shows performance against the system's own best. Shaded bands are sustained shortfalls, green lines are heavy rain, and the crosses are days of zero output in good sun.

**App, open "outage or fault", then "soiling" (2:05 to 2:25)**

Under likely causes, facts and hypotheses are kept apart. Outage: twenty-nine days of near-zero output, which is a fact. Soiling: tested, and not distinguishable from scatter, so the tool says so instead of guessing.

## Part C: the report (2:25 to 2:50)

**report_JAIPUR1.pdf, page 1 then page 2**

Each system also gets a two-page report that an owner, installer or lender can audit: what was measured, what was estimated, how sure, and the limits. It is physics and statistics only. No machine learning and no language model is used in the analysis.

## Part D: slides (2:50 to 4:00)

**Slide 6 (2:50 to 3:25)**

What can I honestly claim today? An estimated loss, on public data. Most of the Jaipur loss is that December breakdown, which the dataset's authors also report, so the tool found it independently. I tested the method by injecting a known loss into four years of data from three systems in Las Vegas: it recovered eighty-five percent, inside its stated range. On one year of Jaipur data that check did not pass, so those figures are coarse. A real cleaning test, before and after, is planned and not yet done.

**Slide 7 (3:25 to 3:50)**

The first users are small commercial rooftop owners and their installers. It is software only and needs no new hardware where the inverter already records generation. The business model is still a hypothesis. My ask is mentorship, pilot rooftops and data partners, so the next result is a measured recovery on an Indian roof.

**Slide 8 (3:50 to 4:00)**

Every figure here traces to a source. Thank you.

## Before recording

1. Run `.venv\Scripts\python -m trueyield run` once and leave the terminal showing the finished output. Do not run it live: it takes about two minutes.
2. Start the app with `.venv\Scripts\python -m streamlit run app.py`, open it in a browser, select JAIPUR1 and scroll to the top.
3. Open `results/report_JAIPUR1.pdf`.
4. Open the deck in slideshow mode.
5. Set the browser zoom so the three headline numbers are large and readable at 720p.

## Optional: slides 4 and 5 (about 15 seconds each)

These are not in the 4-minute timings above. Use them only if a read-through leaves you time, between slide 3 and the demo. The same text is in the deck's speaker notes.

**Slide 4.** The architecture is a straight pipeline: clean the inverter data, model the expected yield, estimate the loss with a range, suggest a likely cause, and verify recovery around a dated event. It is built in Python with pvlib and NASA weather data.

**Slide 5.** Solar monitoring already exists, sold by installers and hardware makers. What is missing is a neutral record of lost and recovered energy that says how sure it is. That is what I add.
