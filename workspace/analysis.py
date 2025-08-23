import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
PLOT_DIR = RESULTS_DIR / "plots"

# Create directories if they don't exist
RESULTS_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

# Plot configuration
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
FIGURE_SIZE = (12, 8)
TOP_N = 9

def load_data():
    """Load flight and weather data from CSV files."""
    logger.info("Loading flight data...")
    
    try:
        flights2022 = pd.read_csv("flights2022.csv")
        flights_weather2022 = pd.read_csv("flights_weather2022.csv")
        
        logger.info(f"Loaded {len(flights2022)} flight records")
        logger.info(f"Loaded {len(flights_weather2022)} weather records")
        
        return flights2022, flights_weather2022
    
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_data(flights_df):
    """Create derived features for analysis."""
    logger.info("Preprocessing flight data...")
    
    # Creating route column
    flights_df["route"] = flights_df["origin"] + "-" + flights_df["dest"]
    
    logger.info(f"Created {flights_df['route'].nunique()} unique routes")
    return flights_df

def analyze_routes(flights_df):
    """Analyze delays and cancellations by route."""
    logger.info("Analyzing routes...")
    
    # Calculate mean departure delay and number of canceled flights for each unique flight route
    routes_delays_cancels = flights_df.groupby("route").agg(
        mean_dep_delay=("dep_delay", "mean"),
        total_cancellations=("dep_time", lambda x: x.isna().sum())
    ).reset_index()
    
    # Identify routes with the highest mean departure delays
    top_routes_by_delay = routes_delays_cancels.sort_values("mean_dep_delay", ascending=False).head(TOP_N)
    
    # Identify routes with the highest number of cancellations
    top_routes_by_cancellations = routes_delays_cancels.sort_values("total_cancellations", ascending=False).head(TOP_N)
    
    logger.info(f"Top route by delays: {top_routes_by_delay.iloc[0]['route']} "
                f"({top_routes_by_delay.iloc[0]['mean_dep_delay']:.1f} min)")
    logger.info(f"Top route by cancellations: {top_routes_by_cancellations.iloc[0]['route']} "
                f"({top_routes_by_cancellations.iloc[0]['total_cancellations']} cancellations)")
    
    return routes_delays_cancels, top_routes_by_delay, top_routes_by_cancellations

def analyze_airlines(flights_df):
    """Analyze delays and cancellations by airline."""
    logger.info("Analyzing airlines...")
    
    # Finding mean departure delays and total cancellations by airline
    airlines_delays_cancels = flights_df.groupby("airline").agg(
        mean_dep_delay=("dep_delay", "mean"),
        total_cancellations=("dep_time", lambda x: x.isna().sum())
    ).reset_index()
    
    # Identify airlines with the highest mean departure delay
    top_airlines_by_delay = airlines_delays_cancels.sort_values("mean_dep_delay", ascending=False).head(TOP_N)
    
    # Identify airlines with the highest number of cancellations
    top_airlines_by_cancellations = airlines_delays_cancels.sort_values("total_cancellations", ascending=False).head(TOP_N)
    
    logger.info(f"Top airline by delays: {top_airlines_by_delay.iloc[0]['airline']} "
                f"({top_airlines_by_delay.iloc[0]['mean_dep_delay']:.1f} min)")
    
    return airlines_delays_cancels, top_airlines_by_delay, top_airlines_by_cancellations

def analyze_wind_impact(flights_weather_df):
    """Analyze the impact of wind conditions on departure delays."""
    logger.info("Analyzing wind impact on delays...")
    
    # Group by wind conditions
    flights_weather_df["wind_group"] = flights_weather_df["wind_gust"].apply(
        lambda x: ">= 10mph" if x >= 10 else "< 10mph"
    )
    
    wind_grouped_data = flights_weather_df.groupby(["wind_group", "origin"]).agg(
        mean_dep_delay=("dep_delay", "mean")
    )
    
    # Summary statistics
    wind_summary = flights_weather_df.groupby("wind_group")["dep_delay"].agg(['mean', 'count'])
    
    print("\n" + "="*50)
    print("WIND IMPACT ANALYSIS")
    print("="*50)
    print("\nSummary by Wind Conditions:")
    print(wind_summary)
    print("\nDetailed by Origin Airport:")
    print(wind_grouped_data)
    
    return wind_grouped_data, wind_summary

def create_route_cancellations_plot(top_routes_by_cancellations):
    """Create bar plot for routes with highest cancellations."""
    logger.info("Creating route cancellations plot...")
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    bars = ax.bar(range(len(top_routes_by_cancellations)), 
                  top_routes_by_cancellations["total_cancellations"],
                  color='red', alpha=0.7)
    
    ax.set_xlabel("Route", fontsize=12)
    ax.set_ylabel("Total Cancellations", fontsize=12)
    ax.set_title("Routes with Highest Number of Cancellations", fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(top_routes_by_cancellations)))
    ax.set_xticklabels(top_routes_by_cancellations["route"], rotation=45, ha='right')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "route_cancellations.png", dpi=300, bbox_inches='tight')
    plt.show()

def create_airline_delays_plot(top_airlines_by_delay):
    """Create bar plot for airlines with highest mean delays."""
    logger.info("Creating airline delays plot...")
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    bars = ax.bar(range(len(top_airlines_by_delay)), 
                  top_airlines_by_delay["mean_dep_delay"],
                  color='orange', alpha=0.7)
    
    ax.set_xlabel("Airline", fontsize=12)
    ax.set_ylabel("Mean Departure Delay (minutes)", fontsize=12)
    ax.set_title("Airlines with Highest Mean Departure Delays", fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(top_airlines_by_delay)))
    ax.set_xticklabels(top_airlines_by_delay["airline"], rotation=45, ha='right')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "airline_delays.png", dpi=300, bbox_inches='tight')
    plt.show()

def create_wind_impact_plot(wind_summary):
    """Create visualization for wind impact on delays."""
    logger.info("Creating wind impact plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    wind_groups = wind_summary.index
    mean_delays = wind_summary['mean']
    counts = wind_summary['count']
    
    bars = ax.bar(wind_groups, mean_delays, color=['skyblue', 'coral'], alpha=0.7)
    
    ax.set_xlabel("Wind Conditions", fontsize=12)
    ax.set_ylabel("Mean Departure Delay (minutes)", fontsize=12)
    ax.set_title("Impact of Wind Conditions on Flight Delays", fontsize=14, fontweight='bold')
    
    # Add value labels and sample sizes
    for i, (bar, delay, count) in enumerate(zip(bars, mean_delays, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{delay:.1f} min\n(n={count})', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "wind_impact.png", dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_report(routes_stats, airlines_stats, wind_summary):
    """Generate a summary report of the analysis."""
    logger.info("Generating summary report...")
    
    report = []
    report.append("="*60)
    report.append("FLIGHT DELAY AND CANCELLATION ANALYSIS REPORT")
    report.append("="*60)
    report.append("")
    
    # Route analysis
    report.append("TOP FINDINGS:")
    report.append("-" * 20)
    worst_route = routes_stats[1].iloc[0]  # top_routes_by_delay
    report.append(f"• Worst route for delays: {worst_route['route']} ({worst_route['mean_dep_delay']:.1f} min)")
    
    worst_cancel_route = routes_stats[2].iloc[0]  # top_routes_by_cancellations
    report.append(f"• Route with most cancellations: {worst_cancel_route['route']} ({worst_cancel_route['total_cancellations']} flights)")
    
    # Airline analysis
    worst_airline = airlines_stats[1].iloc[0]  # top_airlines_by_delay
    report.append(f"• Airline with highest delays: {worst_airline['airline']} ({worst_airline['mean_dep_delay']:.1f} min)")
    
    # Wind impact
    report.append(f"• Wind impact: High winds (≥10mph) cause {wind_summary.loc['>= 10mph', 'mean']:.1f} min delays")
    report.append(f"                Low winds (<10mph) cause {wind_summary.loc['< 10mph', 'mean']:.1f} min delays")
    
    report_text = "\n".join(report)
    print("\n" + report_text)
    
    # Save report to file
    with open(RESULTS_DIR / "analysis_summary.txt", 'w') as f:
        f.write(report_text)
    
    logger.info(f"Summary report saved to {RESULTS_DIR / 'analysis_summary.txt'}")

def main():
    """Main analysis pipeline."""
    logger.info("Starting flight analysis...")
    
    try:
        # Load and preprocess data
        flights_df, flights_weather_df = load_data()
        flights_df = preprocess_data(flights_df)
        
        # Perform analyses
        routes_stats = analyze_routes(flights_df)
        airlines_stats = analyze_airlines(flights_df)
        wind_data, wind_summary = analyze_wind_impact(flights_weather_df)
        
        # Create visualizations
        create_route_cancellations_plot(routes_stats[2])  # top_routes_by_cancellations
        create_airline_delays_plot(airlines_stats[1])     # top_airlines_by_delay
        create_wind_impact_plot(wind_summary)
        
        # Generate summary report
        generate_summary_report(routes_stats, airlines_stats, wind_summary)
        
        logger.info("Analysis completed successfully!")
        logger.info(f"Results saved to: {RESULTS_DIR}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()