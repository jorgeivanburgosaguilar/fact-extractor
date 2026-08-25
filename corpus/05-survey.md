What the automated-mooring literature actually measures
Since 2019 a series of trials at container terminals along the eastern seaboard has stopped asking whether automated mooring works in principle and started measuring what happens when it replaces a line-handling crew during a live berthing: how long the approach takes, how often the system aborts, and how tug dispatch changes once mooring time becomes predictable rather than variable. The primary sources are:

Bracknell, Osei and Farr (Halvard Institute), "Vacuum Mooring Under Load: 214 Berthings at Sandhaven Container Terminal" (2023) - the largest single-site trial, two full winters.
Ferreira and Lindqvist (Cauldrick Maritime Institute), "Tug Dispatch After Automated Mooring" (2024) - models tug standby counts against mooring predictability.
Solberg, Achterberg, Nnamdi and Reyes (University of Drommel), "When the Pads Do Not Seat" (2024) - forty-one abort events, coded into six failure classes.
Okonjo and Bracknell (Marine Research Council), "Crew Trust After the First Season" (2024) - interviews with twenty-six line handlers and pilots.
The Institute for Port Automation, "Cross-Terminal Comparison of Mooring Assist Systems" (2025) - vacuum pads against magnetic heads, at three terminals.
Achterberg and Solberg, "Wind Limits and Abort Thresholds" (2023) - abort rate against crosswind speed at approach.
Reyes, Ferreira and Nnamdi, "What the Dispatcher Actually Does" (2024) - a field study of the same dispatch decisions.

1. What the trials measure, and what they do not
The Sandhaven trial is the baseline the rest of the literature works against. Bracknell, Osei and Farr logged pad-seat time, full mooring time and abort rate across 214 berthings between November and March, and found a median mooring time of 6.4 minutes against 14.1 minutes for a conventional four-line crew, with the gap narrowest in calm conditions and widest above a force 5 crosswind. Sandhaven kept its full crew on standby throughout, so the reported time saving is not a labour saving, a distinction later cost-model papers routinely drop.

2. Why pads fail to seat
Solberg, Achterberg, Nnamdi and Reyes reviewed camera and load-cell logs from all forty-one aborted approaches at Sandhaven and Ravensmouth and sorted each into one of six failure classes, tabulated below with the share of aborts each class accounted for and the typical recovery action the crew took.

Failure class	What the log shows	Share of aborts	Typical recovery
Hull fouling	Marine growth on the plating prevents a full seal	34%	Re-approach after a manual scrape
Excessive roll	Pad detaches as the hull rolls past the seal tolerance	27%	Wait for a calmer window
Approach angle	Vessel arrives outside the seven-degree capture cone	20%	Pilot repositions and re-attempts
Pad wear	A worn seal cannot hold vacuum under load	10%	Swap to the reserve pad bank
Power interruption	Terminal-side power dips below the pump threshold	7%	Switch to the backup generator
Sensor fault	The seating sensor reports contact that load cells contradict	2%	Manual crew mooring for that berth

3. What changes for tug dispatch
Ferreira and Lindqvist modelled dispatch decisions before and after automated mooring using two years of scheduling logs from Sandhaven and found that predictable mooring time let the terminal cut its standby tug count from four to three without any increase in waiting time, because dispatchers could schedule the third tug's jobs against a six-minute window instead of a guess between four and twenty minutes. Reyes, Ferreira and Nnamdi's field study of the same dispatchers found the saving came almost entirely from fewer defensive holds - a tug kept idle just in case - which fell from nineteen holds a month to four.

4. What the literature agrees changes, and what it does not
Faster median mooring time under calm conditions is the effect every trial reports and the one measured most consistently, from Sandhaven's 6.4 versus 14.1 minutes to the Institute for Port Automation's smaller but still significant gap at all three of its comparison terminals.
A widening gap between calm and rough conditions rather than a uniform saving is what Achterberg and Solberg actually found: abort rate rises from 3% below a force 4 crosswind to 22% above force 6, so the time saved is concentrated in exactly the conditions that already berthed quickly.
Fewer standby tugs once dispatch is scheduled against a predictable window is Ferreira and Lindqvist's finding, and the only tug-side result any other paper replicates, at Ravensmouth with a similar three-to-two reduction.
No measured reduction in line handler headcount appears at any trial site, because every terminal that tested automated mooring kept its full crew on standby throughout the trial period, which every paper in this list treats as a policy choice rather than a technical finding.
A trust cost in the first season that the numbers alone do not show is what Okonjo and Bracknell's interviews found: nine of twenty-six line handlers described the system as something they were watching in case it failed rather than operating, even after a season with no injury and only the recorded aborts.
Abort handling still runs through the human crew in every recorded case, because none of the forty-one aborts in Solberg, Achterberg, Nnamdi and Reyes's taxonomy resolved automatically; every one ended with a manual re-approach, a wait, or a full manual mooring.
A single head-to-head comparison exists, and it favours vacuum pads on speed but magnetic heads on reliability: the Institute for Port Automation found magnetic mooring aborted in only 4% of approaches against vacuum mooring's 9% across the same three terminals, at a median mooring time about ninety seconds slower.
Disagreement on what predictable time is worth runs through the cost papers: the Halvard Institute's occupancy-minute model prices the saved six to eight minutes per berthing at the terminal's marginal berth-hour rate, a method Reyes, Ferreira and Nnamdi's field study argues overstates the benefit because dispatchers do not actually rebook that time at the margin.
No trial in this literature runs longer than two seasons, so nothing yet speaks to pad wear, seal degradation, or whether the first-season trust cost fades once a crew has watched the system through a full winter.
Abort rate itself falls with experience even inside a single trial: Bracknell, Osei and Farr report Sandhaven's rate dropping from 24% in the first winter to 15% in the second, which every later paper attributes to pilots learning the capture envelope rather than to any change in the equipment.
Pilot certification time looks consistent across sites even where abort rate does not: median supervised approaches before independent use was eleven at Sandhaven, and the Institute for Port Automation's three-terminal comparison reports a similar range of eight to fourteen.
The reserve pad bank that recovers a pad-wear abort needs replacing roughly every four hundred berthings at Sandhaven's usage rate, a maintenance cost that appears in none of the occupancy-minute cost models this literature otherwise relies on.
Ravensmouth's numbers track Sandhaven's closely enough that Ferreira and Lindqvist treat the two sites as one pooled sample for the tug-dispatch model, the one methodological choice every replication study in this list has since copied, though neither site publishes the raw approach-by-approach logs the pooling is based on, so nobody outside the original team can check the pooling itself.

5. What is still open
None of the seven sources answers the question a terminal operator actually asks before buying the system: whether the mooring-time saving survives a third season once pads have worn. The Institute for Port Automation's cross-terminal comparison is the design the field needs more of - matched terminals, one failure taxonomy, a reliability number beside the speed number - but three terminals over one season is a start, not an answer.
