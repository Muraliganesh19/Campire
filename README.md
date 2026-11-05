# Campire
Campire: Camera + Umpire. This is major project topic that I selected to build as part of AI&amp;ML UG program syllabus.
# Introduction
	Amateur and small-ground cricket leagues represent the heart of the sport, yet they lack access to the high-precision officiating tools used in elite matches.
	Addressing the Disparity: Expensive setups like Hawk-Eye remain financially out of reach for grassroots organizations.
	Al-Powered Officiating: Implementing a low-cost, multi-camera system leveraging Computer Vision and Audio analysis.
	Real-Time Decisions: Automating detection for Wides, No- Balls, and Bowler Fouls with machine precision.

# Simulation Data
The AI stages are still in development, so the app is tested with hand-annotated deliveries stored in `simulation/`. This folder is not committed to git because of its size.

Each bowling delivery is recorded by 4 cameras, which gives 4 clips. All 4 clips of a delivery are kept in one `type*` folder, and each clip has its own `c*` folder:

```text
simulation/
├── ip_vid/                      # annotated deliveries
│   ├── type1/                   # one delivery
│   │   ├── c1/                  # clip from camera 1
│   │   │   ├── images/          # frame_000000.PNG, frame_000001.PNG, ...
│   │   │   └── annotations.xml  # labels for this clip
│   │   ├── c2/                  # same layout as c1
│   │   ├── c3/
│   │   ├── c4/
│   │   └── decisions.json       # decisions for this delivery
│   ├── type2/
│   └── type3/ ...
└── op_vid/                      # output videos
```

- **`images/`:** PNG frames, numbered from `frame_000000.PNG`. Frame count and resolution can differ between the clips of one delivery.
- **`annotations.xml`:** a CVAT 1.1 video export with one track per annotated object, drawn as boxes and ellipses. The labels are `ball`, `wicket`, `batter`, `bowler`, `bowler-front`, `bowler-back`, `bowler-left`, `bowler-right`, `wide-front`, `wide-left` and `wide-right`.
- **`decisions.json`:** the decisions the host app displays for the delivery. It must not be empty:

```json
{
  "bowler_stance": "LEGAL",
  "delivery": "WIDE",
  "wicket": "NOT-HIT"
}
```
