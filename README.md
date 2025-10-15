# 📈 Competitive Investment Portfolio Dashboard

A modern, professional dashboard to visualize and compare investment portfolios of multiple individuals with real-time data, forecasting, and performance analytics.

## 🌟 Features

- **Real-time Data**: Fetches live stock and cryptocurrency data from Yahoo Finance
- **Trend Visualization**: Interactive charts showing price movements from October 1st to November 13th
- **Forecasting Engine**: Uses linear regression and exponential smoothing for price predictions
- **Performance Ranking**: Dynamic table with color-coded performance metrics
- **Professional UI**: Modern design with smooth animations and responsive layout
- **Portfolio Overview**: Aggregate statistics and performance insights
- **Auto-refresh**: Cached data with 5-minute refresh intervals

## 👥 Portfolio Configuration

The dashboard tracks 4 investors and their respective assets:

| Investor | Symbol | Company/Asset | Badge |
|----------|---------|---------------|-------|
| M Maard | UUU | Energy Fuels Inc | MM |
| Ohmmy | CCJ | Cameco Corp | OH |
| Blnm | DELTA.BK | Delta Electronics (Thailand) PCL | BL |
| NWiz | BTC-USD | Bitcoin | NW |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the repository**
   ```bash
   cd portfolio_dashboard_demo
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```

4. **Access the dashboard**
   - Open your browser and go to `http://localhost:8501`
   - The dashboard will automatically start fetching data

## 🎯 Dashboard Components

### 1. Price Trend Charts
- **4 individual charts** for each portfolio
- **Historical data** from October 1st to current date
- **Forecast curves** projecting to November 13th
- **Interactive tooltips** with hover values
- **Professional styling** with smooth animations

### 2. Performance Ranking Table
- **Dynamic sorting** by current performance
- **Color coding**: Green for gains, red for losses
- **Comparison metrics**:
  - Current vs. October 13/14 baseline
  - Forecast price for November 13th
  - Projected performance percentage

### 3. Portfolio Overview
- **Average performance** across all portfolios
- **Best/worst performers** identification
- **Forecast insights** and trends
- **Key statistics** at a glance

### 4. Interactive Controls
- **Refresh button** for latest data
- **Date range selector** for custom periods
- **Display options** for forecasts and overview
- **Responsive design** for all screen sizes

## 🔧 Technical Architecture

### Data Sources
- **Yahoo Finance API**: Stock data via `yfinance` library
- **Real-time updates**: 5-minute cache refresh
- **Error handling**: Graceful fallbacks for API issues

### Forecasting Models
- **Linear Regression**: Trend-based predictions
- **Exponential Smoothing**: Adaptive forecasting
- **Combined Approach**: Weighted average of both methods

### Frontend Technology
- **Streamlit**: Web application framework
- **Plotly**: Interactive charts and visualizations
- **Custom CSS**: Professional styling and animations
- **Responsive Design**: Mobile and desktop compatible

## 📊 Metrics & Calculations

### Performance Metrics
- **Baseline Date**: October 13th (October 14th for DELTA)
- **Current Performance**: `((Current Price - Baseline Price) / Baseline Price) × 100`
- **Forecast Performance**: `((Forecast Price - Baseline Price) / Baseline Price) × 100`

### Forecasting Algorithm
1. **Data Preparation**: Historical price normalization
2. **Linear Regression**: Trend line calculation
3. **Exponential Smoothing**: Adaptive trend analysis
4. **Weighted Combination**: 60% linear + 40% exponential
5. **Date Projection**: 30-day forward prediction

## 🎨 Customization

### Styling
- **Color Scheme**: Modern gradient themes
- **Typography**: Inter font family
- **Animations**: Smooth hover effects
- **Layout**: Professional card-based design

### Configuration
- **Portfolio data**: Easily modify symbols and names in `PORTFOLIOS` dictionary
- **Date ranges**: Adjustable baseline and forecast periods
- **Refresh intervals**: Configurable cache timing
- **Display options**: Toggle forecasts and overview sections

## 📱 Responsive Design

The dashboard is fully responsive and works on:
- **Desktop**: Full feature experience
- **Tablet**: Optimized layout
- **Mobile**: Touch-friendly interface

## 🔄 Data Refresh

- **Automatic**: Every 5 minutes via Streamlit caching
- **Manual**: Click the "Refresh Data" button
- **Progress Indicators**: Loading states for better UX

## 📈 Performance Monitoring

The dashboard includes:
- **Error handling** for API failures
- **Loading indicators** for data fetching
- **Performance metrics** for quick insights
- **Data validation** for accuracy

## 🛠️ Development

### File Structure
```
portfolio_dashboard_demo/
├── app.py              # Main application
├── requirements.txt    # Dependencies
└── README.md          # Documentation
```

### Adding New Portfolios
1. Update the `PORTFOLIOS` dictionary in `app.py`
2. Add symbol, company name, and badge
3. Restart the application

### Modifying Forecasts
- Adjust weights in the `calculate_forecast()` function
- Modify the forecasting horizon (default: 30 days)
- Add additional forecasting models

## 🔒 Security & Privacy

- **No API keys required**: Uses free Yahoo Finance data
- **Local processing**: All calculations run locally
- **No data storage**: No user data persistence
- **Privacy focused**: No external tracking

## 📞 Support

For issues or questions:
1. Check the console output for error messages
2. Verify internet connection for data fetching
3. Ensure all dependencies are properly installed
4. Restart the application if issues persist

## 🚀 Future Enhancements

Potential additions:
- **Export functionality**: CSV/PDF report generation
- **Historical volatility**: Risk analysis indicators
- **Email alerts**: Performance notifications
- **Portfolio optimization**: Allocation suggestions
- **More data sources**: Additional market APIs
- **Advanced forecasting**: ML-based predictions

---

**Built with ❤️ using Streamlit, Plotly, and modern web technologies**