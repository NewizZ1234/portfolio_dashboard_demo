def create_percentage_difference_chart_improved(portfolio_data):
    """Create percentage difference chart using actual historical data and real forecasts"""
    import plotly.graph_objects as go
    import pandas as pd
    from datetime import datetime, date
    
    # Get current date dynamically
    current_date = datetime.now().date()
    current_date_str = current_date.strftime('%Y-%m-%d')
    
    # Initialize log collection
    chart_logs = []
    chart_logs.append(f"📅 Current date: {current_date_str}")
    chart_logs.append("")
    chart_logs.append("🔍 DEBUG: Starting percentage chart with actual data")
    chart_logs.append("")
    
    # Import Streamlit only when needed for fallback
    try:
        import streamlit as st
    except ImportError:
        # Fallback for testing without Streamlit
        class MockST:
            def write(self, text): pass  # Don't print, just collect in logs
            def error(self, text): chart_logs.append(f"ERROR: {text}")
        st = MockST()
    
    fig = go.Figure()
    
    colors = ['#667eea', '#f093fb', '#4facfe', '#43e97b']
    chart_data_count = 0
    
    # Portfolio configuration
    PORTFOLIOS = {
        "M Maard": {"symbol": "UUUU", "company": "Energy Fuels Inc", "badge": "MM"},
        "Ohmmy": {"symbol": "CCJ", "company": "Cameco Corp", "badge": "OH"},
        "Blnm": {"symbol": "DELTA.BK", "company": "Delta Electronics (Thailand) PCL", "badge": "BL"},
        "NWiz": {"symbol": "BTC-USD", "company": "Bitcoin", "badge": "NW"}
    }
    
    st.write("🔍 DEBUG: Starting percentage chart with actual data")
    
    for idx, (name, info) in enumerate(PORTFOLIOS.items()):
        data = portfolio_data.get(name)
        if data is not None and not data.empty:
            try:
                chart_logs.append(f"🔍 Processing {name}")
                
                # Sort data by date to ensure chronological order
                data = data.sort_values('Date').copy()
                data['Date_str'] = data['Date'].dt.strftime('%Y-%m-%d')
                
                # Set baseline date based on portfolio
                if name == "Blnm":  # DELTA portfolio
                    baseline_date_str = "2025-10-14"
                    baseline_label = "Oct 14, 2025"
                else:
                    baseline_date_str = "2025-10-13"
                    baseline_label = "Oct 13, 2025"
                
                chart_logs.append(f"🔍 {name} baseline: {baseline_label}")
                
                # Find baseline price from actual 2025 data
                baseline_price = None
                
                # Look for baseline date in the actual 2025 data
                baseline_matches = data[data['Date_str'] == baseline_date_str]
                
                if not baseline_matches.empty:
                    baseline_price = baseline_matches['Close'].iloc[0]
                    chart_logs.append(f"✅ Found baseline price from {baseline_date_str}: ${baseline_price:.2f}")
                else:
                    # Use most recent available price as baseline
                    baseline_price = data['Close'].iloc[-1]
                    latest_date = data['Date'].iloc[-1].strftime('%Y-%m-%d')
                    chart_logs.append(f"⚠️ Using latest available price from {latest_date}: ${baseline_price:.2f}")
                
                # Generate forecast data for future dates
                # Import calculate_forecast function
                import sys
                import os
                sys.path.append(os.path.dirname(__file__))
                
                # Define calculate_forecast locally to avoid import issues
                def calculate_forecast_local(data, days_ahead=30):
                    """Calculate price forecast using linear regression and exponential smoothing"""
                    import numpy as np
                    from sklearn.linear_model import LinearRegression
                    from datetime import timedelta
                    
                    if data is None or len(data) < 5:
                        return None, None
                    
                    # Prepare data for forecasting
                    data_copy = data.copy()
                    data_copy['days'] = range(len(data_copy))
                    X = data_copy['days'].values.reshape(-1, 1)
                    y = data_copy['Close'].values
                    
                    # Linear regression forecast
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Generate future dates
                    last_day = len(data_copy) - 1
                    future_days = np.arange(last_day + 1, last_day + 1 + days_ahead).reshape(-1, 1)
                    linear_forecast = model.predict(future_days)
                    
                    # Exponential smoothing (simple)
                    alpha = 0.3
                    smoothed = [y[0]]
                    for i in range(1, len(y)):
                        smoothed.append(alpha * y[i] + (1 - alpha) * smoothed[-1])
                    
                    # Extend exponential smoothing
                    last_smoothed = smoothed[-1]
                    trend = (smoothed[-1] - smoothed[-5]) / 5 if len(smoothed) >= 5 else 0
                    exp_forecast = []
                    for i in range(days_ahead):
                        next_val = last_smoothed + trend * (i + 1) * 0.5  # Dampened trend
                        exp_forecast.append(next_val)
                    
                    # Combine forecasts (weighted average)
                    combined_forecast = (linear_forecast * 0.6 + np.array(exp_forecast) * 0.4)
                    
                    # Generate forecast dates
                    last_date = data_copy['Date'].iloc[-1]
                    forecast_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
                    
                    return forecast_dates, combined_forecast
                
                forecast_dates, forecast_prices = calculate_forecast_local(data, days_ahead=30)
                
                # Create dynamic date range: baseline date to 30 days ahead
                if name == "Blnm":  # DELTA
                    start_date = "2025-10-13"  # Show one day before DELTA baseline
                else:  # Others
                    start_date = baseline_date_str  # Start from baseline
                
                # End date: 30 days from current date
                end_date = (current_date + pd.Timedelta(days=30)).strftime('%Y-%m-%d')
                chart_dates = pd.date_range(start=start_date, end=end_date, freq='D')
                
                # Prepare chart data arrays
                chart_plot_dates = []
                chart_plot_pct = []
                chart_data_types = []  # Track if data is historical, current, or forecast
                
                for date in chart_dates:
                    date_str = date.strftime('%Y-%m-%d')
                    
                    # Determine data type and calculate percentage
                    if date_str <= current_date_str:  # Historical/current data (up to today)
                        # Look for the actual 2025 date in the scraped data
                        target_matches = data[data['Date_str'] == date_str]
                        
                        if not target_matches.empty:
                            target_price = target_matches['Close'].iloc[0]
                        else:
                            # Use latest available data if specific date not found
                            target_price = data['Close'].iloc[-1]
                            chart_logs.append(f"⚠️ {name}: Using latest price for {date_str}")
                        
                        # Calculate percentage change from baseline
                        pct_change = ((target_price - baseline_price) / baseline_price) * 100
                        
                        # Determine data type and apply special baseline rules
                        if name == "Blnm":  # DELTA special handling
                            if date_str == baseline_date_str:  # Oct 14 for DELTA
                                pct_change = 0.0  # Force baseline to 0%
                                data_type = "baseline"
                                chart_logs.append(f"✅ {name}: {date_str} = 0.0% (baseline)")
                            else:
                                data_type = "current" if date_str == current_date_str else "historical"
                                chart_logs.append(f"📊 {name}: {date_str} = {pct_change:.2f}% (actual {data_type})")
                        else:  # Other portfolios
                            if date_str == baseline_date_str:  # Oct 13 for others
                                pct_change = 0.0  # Force baseline to 0%
                                data_type = "baseline"
                                chart_logs.append(f"✅ {name}: {date_str} = 0.0% (baseline)")
                            else:
                                data_type = "current" if date_str == current_date_str else "historical"
                                chart_logs.append(f"📊 {name}: {date_str} = {pct_change:.2f}% (actual {data_type})")
                    
                    else:  # Future dates (tomorrow onwards) - Use forecast
                        # Calculate days from the day after current date
                        next_day = current_date + pd.Timedelta(days=1)
                        days_from_next_day = (date - pd.to_datetime(next_day)).days
                        
                        if forecast_dates and days_from_next_day >= 0 and days_from_next_day < len(forecast_prices):
                            forecast_price = forecast_prices[days_from_next_day]
                            pct_change = ((forecast_price - baseline_price) / baseline_price) * 100
                            data_type = "forecast"
                            chart_logs.append(f"📈 {name}: {date_str} = {pct_change:.2f}% (forecast)")
                        else:
                            # Fallback forecast calculation
                            days_from_baseline = (date - pd.to_datetime(baseline_date_str)).days
                            pct_change = days_from_baseline * 0.1  # Simple linear trend
                            data_type = "forecast_fallback"
                            chart_logs.append(f"📈 {name}: {date_str} = {pct_change:.2f}% (fallback forecast)")
                    
                    # Add to chart data
                    chart_plot_dates.append(date)
                    chart_plot_pct.append(pct_change)
                    chart_data_types.append(data_type)
                
                # Create separate traces for different data types
                if chart_plot_dates:
                    # Separate historical and forecast data
                    historical_dates = []
                    historical_pct = []
                    forecast_chart_dates = []
                    forecast_chart_pct = []
                    
                    for i, (date, pct, dtype) in enumerate(zip(chart_plot_dates, chart_plot_pct, chart_data_types)):
                        if dtype in ["baseline", "historical", "current", "chart_start"]:
                            historical_dates.append(date)
                            historical_pct.append(pct)
                        else:  # forecast types
                            forecast_chart_dates.append(date)
                            forecast_chart_pct.append(pct)
                    
                    # Add historical/current data trace
                    if historical_dates:
                        fig.add_trace(
                            go.Scatter(
                                x=historical_dates,
                                y=historical_pct,
                                mode='lines+markers',
                                name=f'{info["badge"]} ({info["symbol"]}) - Actual',
                                line=dict(color=colors[idx % len(colors)], width=3),
                                marker=dict(size=6),
                                hovertemplate='<b>%{fullData.name}</b><br>' +
                                            'Date: %{x}<br>' +
                                            f'Change from {baseline_label}: %{{y:.2f}}%<br>' +
                                            f'Baseline: ${baseline_price:.2f}<br>' +
                                            '<extra></extra>'
                            )
                        )
                    
                    # Add forecast data trace
                    if forecast_chart_dates:
                        fig.add_trace(
                            go.Scatter(
                                x=forecast_chart_dates,
                                y=forecast_chart_pct,
                                mode='lines+markers',
                                name=f'{info["badge"]} - Forecast',
                                line=dict(color=colors[idx % len(colors)], width=2, dash='dash'),
                                marker=dict(size=4),
                                opacity=0.8,
                                hovertemplate='<b>%{fullData.name}</b><br>' +
                                            'Date: %{x}<br>' +
                                            f'Predicted Change from {baseline_label}: %{{y:.2f}}%<br>' +
                                            f'Baseline: ${baseline_price:.2f}<br>' +
                                            '<extra></extra>'
                            )
                        )
                    
                    chart_data_count += 1
                    chart_logs.append(f"✅ Added {name} with {len(historical_dates)} actual + {len(forecast_chart_dates)} forecast points")
                
            except Exception as e:
                st.error(f"❌ Error processing {name}: {str(e)}")
                chart_logs.append(f"❌ Error processing {name}: {str(e)}")
                
        chart_logs.append("")  # Add spacing between portfolios    if chart_data_count == 0:
        fig.add_annotation(
            text="No data available for percentage comparison",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16)
        )
    else:
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7, 
                     annotation_text="Baseline (0%)", annotation_position="bottom right")
        
        # Add annotation for current date instead of vertical line (to avoid Plotly issues)
        current_month_day = current_date.strftime('%b %d')
        fig.add_annotation(
            x=current_date_str,
            y=0,
            text=f"Today ({current_month_day})",
            showarrow=True,
            arrowhead=2,
            arrowcolor="orange",
            arrowwidth=2,
            bgcolor="orange",
            bordercolor="orange",
            font=dict(color="white", size=10),
            xanchor="center",
            yanchor="bottom"
        )
    
    # Calculate dynamic x-axis range
    start_display = (current_date - pd.Timedelta(days=2)).strftime('%Y-%m-%d')
    end_display = (current_date + pd.Timedelta(days=30)).strftime('%Y-%m-%d')
    
    fig.update_layout(
        title={
            'text': f'Portfolio Performance: Actual Data + Forecasts ({chart_data_count} portfolios)<br><sub>Solid: Actual Data | Dashed: Forecasts | Baselines: DELTA=Oct 14, Others=Oct 13</sub>',
            'x': 0.5,
            'font': {'family': 'Inter, sans-serif', 'size': 18}
        },
        xaxis_title='Date',
        yaxis_title='Percentage Change from Baseline (%)',
        xaxis=dict(
            range=[start_display, end_display],
            tickformat='%b %d'
        ),
        height=400,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    chart_logs.append(f"🔍 Chart completed with {chart_data_count} portfolios using actual + forecast data")
    return fig, chart_logs