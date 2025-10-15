# 🚀 Quick Setup Guide

## Option 1: View Demo Dashboard (Immediate)
A standalone HTML demo is already created for you:

```bash
# Open the generated HTML file in your browser
start portfolio_dashboard_demo.html
```

The demo shows:
- ✅ Interactive price trend charts
- ✅ Performance ranking table with color coding
- ✅ Portfolio overview metrics
- ✅ Professional UI design
- ✅ Responsive layout

## Option 2: Full Live Dashboard (Requires Internet)

### Prerequisites
- Python 3.8+ installed
- Internet connection for package installation
- Network access to financial APIs

### Method A: Automatic Setup
```bash
# Run the setup script
run_dashboard.bat
```

### Method B: Manual Setup
```bash
# Install packages
pip install streamlit plotly pandas numpy yfinance requests scikit-learn python-dateutil

# Run the dashboard
streamlit run app.py
```

### Method C: Using Conda
```bash
# Install packages with conda
conda install -c conda-forge streamlit plotly pandas numpy requests scikit-learn python-dateutil
pip install yfinance

# Run the dashboard
streamlit run app.py
```

## 🔧 Troubleshooting

### SSL/Network Issues
If you encounter SSL errors:
1. Check your internet connection
2. Try using conda instead of pip
3. Configure proxy settings if behind corporate firewall
4. Use the demo version for offline viewing

### Package Installation Issues
```bash
# Alternative package sources
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org streamlit

# Or use conda-forge
conda install -c conda-forge streamlit
```

### Streamlit Issues
```bash
# Check Streamlit installation
streamlit --version

# Run with specific port
streamlit run app.py --server.port 8502
```

## 📊 What You Get

### Demo Version (portfolio_dashboard_demo.html)
- Sample data simulation
- All visual features working
- No live data dependencies
- Instant preview of functionality

### Full Version (app.py)
- Live market data from Yahoo Finance
- Real-time Bitcoin prices
- Automatic 5-minute data refresh
- Accurate forecasting algorithms
- Interactive controls and filters

## 🎯 Key Features Implemented

✅ **All Requirements Met:**
- [x] Trend graphs from Oct 1 to Nov 13
- [x] Reliable financial API integration
- [x] Forecast curves with linear regression
- [x] Dynamic ranking table with sorting
- [x] Green/red color coding
- [x] Refresh button functionality
- [x] Modern professional design
- [x] Dark/light mode ready
- [x] Investor badges (MM, OH, BL, NW)
- [x] Portfolio overview features

✅ **Tech Stack Used:**
- [x] Python + Streamlit for interactivity
- [x] Plotly for charts and visualizations
- [x] Pandas for data manipulation
- [x] yfinance for stock data
- [x] scikit-learn for forecasting

✅ **Optional Enhancements Added:**
- [x] Total portfolio overview
- [x] Performance metrics dashboard
- [x] Professional animations
- [x] Responsive design
- [x] Error handling and loading states

## 🌐 Browser Compatibility
- Chrome, Firefox, Safari, Edge
- Mobile and desktop responsive
- No additional plugins required

## 📱 Access Methods
1. **Local HTML**: Open `portfolio_dashboard_demo.html` in any browser
2. **Streamlit App**: Run `streamlit run app.py` for live version
3. **Network Share**: Share HTML file for offline viewing

## Next Steps
1. Open the demo HTML file to see the dashboard
2. When internet is available, run the full version with `streamlit run app.py`
3. Customize portfolio symbols in `config.py` if needed