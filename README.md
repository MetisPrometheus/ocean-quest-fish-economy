# 🐟 Fish Economy Dashboard

A Streamlit app for Roblox fishing game economy balancing.

## Features

- **🔧 Generate Lua** - Upload `.rbxmx` file → generates `FishData.lua` with calibrated economy
- **📊 Zone Analysis** - See gold/hour at every rod level for each zone
- **🎣 Rod Comparison** - Compare all zones for a specific rod level
- **📈 Progression** - Optimal zone for each rod level 1-33
- **🎲 Simulation** - Run mock fishing sessions with RNG
- **📋 Fish Database** - Search and filter all fish

## Live Demo

👉 **[fish-economy.streamlit.app](https://fish-economy.streamlit.app)** _(update with your URL)_

## How to Use

### Online

1. Go to the live app
2. Upload your `.rbxmx` file (exported from Roblox Studio)
3. Download the generated `FishData.lua`
4. Copy into your Roblox game

### Local

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/fish-economy-dashboard.git
cd fish-economy-dashboard

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run fish_dashboard.py
```

## Economy Model

The app uses a Gaussian distribution model for fish drops:

- **Rod progression** shifts the mean toward rarer fish
- **Sigma** (spread) increases with rod level
- **Calibration** ensures each zone is ~2% better than the previous at transition points

### Zone Configuration

| Zone     | Rod Range |
| -------- | --------- |
| Pirate   | 1-6       |
| Greek    | 7-15      |
| Japanese | 16-24     |
| Viking   | 25-33     |

## File Structure

```
fish-economy-dashboard/
├── fish_dashboard.py    # Main Streamlit app
├── requirements.txt     # Dependencies
└── README.md
```

## Screenshots

_(Add screenshots of your app here)_

## License

MIT
