import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_earnings_dates(ticker, start_date):
    stock = yf.Ticker(ticker)
    earnings = stock.earnings_dates
    earnings = earnings[earnings.index >= start_date]
    return earnings.index.strftime('%Y-%m-%d').tolist()

def calculate_percentage_changes(ticker, earnings_dates):
    stock = yf.Ticker(ticker)
    percentage_changes = []
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    for date in earnings_dates:
        try:
            if date > current_date:
                continue  # Skip dates in the future
            
            start_date = datetime.strptime(date, "%Y-%m-%d")
            end_date = start_date + timedelta(days=5)
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty or len(hist) < 2:
                continue
            
            open_price = hist.iloc[0]['Open']
            close_price = hist.iloc[-1]['Close']
            percent_change = ((close_price - open_price) / open_price) * 100
            percentage_changes.append((date, percent_change))
        except Exception as e:
            print(f"Error processing date {date}: {e}")
    
    return percentage_changes

def analyze_changes(percentage_changes):
    gains = [change[1] for change in percentage_changes if change[1] > 0]
    losses = [change[1] for change in percentage_changes if change[1] <= 0]
    
    num_gains = len(gains)
    num_losses = len(losses)
    avg_gain = sum(gains) / num_gains if num_gains > 0 else 0
    avg_loss = sum(losses) / num_losses if num_losses > 0 else 0
    total_avg_change = sum([change[1] for change in percentage_changes]) / len(percentage_changes) if percentage_changes else 0
    
    return {
        "num_gains": num_gains,
        "num_losses": num_losses,
        "avg_gain": avg_gain,
        "avg_loss": avg_loss,
        "total_avg_change": total_avg_change
    }

def main(ticker, start_date):
    
    
    earnings_dates = get_earnings_dates(ticker, start_date)
    percentage_changes = calculate_percentage_changes(ticker, earnings_dates)
    analysis = analyze_changes(percentage_changes)
    
    print(f"Earnings analysis for {ticker} since {start_date}:")
    for date, change in percentage_changes:
        print(f"Earnings Date: {date} - Percentage Change: {change:.2f}%")
    print(f"Number of gains: {analysis['num_gains']}")
    print(f"Number of losses: {analysis['num_losses']}")
    print(f"Average gain: {analysis['avg_gain']:.2f}%")
    print(f"Average loss: {analysis['avg_loss']:.2f}%")
    print(f"Total average change: {analysis['total_avg_change']:.2f}%")

# if __name__ == "__main__":
#     main()
