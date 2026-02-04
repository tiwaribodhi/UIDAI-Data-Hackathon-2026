"""
Aadhaar Demographic Analysis Dashboard & Intelligence System
===========================================================
Comprehensive analysis of Aadhaar demographic data across districts and demographic groups.
Features: Single/multi-district analysis, statistical metrics, risk classification, 
projections, and Pareto analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# =========================
# CONFIGURATION & FILE PATHS
# =========================
BASE_PATH = r"C:\New folder\Program1\Hackathon\CSV Files\api_data_aadhar_demographic"

CSV_FILES = [
    rf"{BASE_PATH}\api_data_aadhar_demographic_0_500000.csv",
    rf"{BASE_PATH}\api_data_aadhar_demographic_500000_1000000.csv",
    rf"{BASE_PATH}\api_data_aadhar_demographic_1000000_1500000.csv",
    rf"{BASE_PATH}\api_data_aadhar_demographic_1500000_2000000.csv",
    rf"{BASE_PATH}\api_data_aadhar_demographic_2000000_2071700.csv"
]

AGE_COLUMNS = ['demo_age_5_17', 'demo_age_17_']

# =========================
# LOAD AND MERGE DATA
# =========================
dataframes = [pd.read_csv(file) for file in CSV_FILES]
df = pd.concat(dataframes, ignore_index=True)

# =========================
# DATA CLEANING
# =========================
def clean_data(data):
    """Clean and standardize data columns."""
    data['district'] = data['district'].astype(str).str.strip().str.lower()
    data['pincode'] = data['pincode'].astype(str).str.strip()
    data[AGE_COLUMNS] = data[AGE_COLUMNS].apply(pd.to_numeric, errors='coerce').fillna(0)
    data['total_demographic'] = data[AGE_COLUMNS].sum(axis=1)
    return data

df = clean_data(df)

# =========================
# PRECOMPUTE CORE AGGREGATES
# =========================
district_group = df.groupby('district')
pincode_group = df.groupby(['district', 'pincode'])

district_total = district_group['total_demographic'].sum()
age_summary = district_group[AGE_COLUMNS].sum()
pincode_count = district_group['pincode'].nunique()
pincode_sum = pincode_group['total_demographic'].sum().reset_index()

# =========================
# RISK CLASSIFICATION & CLUSTERING SYSTEM
# =========================

def calculate_risk_metrics():
    """Calculate comprehensive risk metrics for all districts."""
    
    # Intensity Index: demographic pressure per pincode
    intensity_index = (district_total / pincode_count).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Child Inclusion Ratio: demographic ratio between age groups
    child_ratio = (
        age_summary['demo_age_5_17'] / age_summary['demo_age_17_'].replace(0, np.nan)
    ).fillna(0)
    
    # Concentration Index (CV): inequality across pincodes
    cv_df = pincode_sum.groupby('district')['total_demographic'].agg(['mean', 'std'])
    cv_df['concentration_index'] = (
        cv_df['std'] / cv_df['mean']
    ).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Risk Score
    risk_df = pd.DataFrame({
        'total_demographic': district_total,
        'child_ratio': child_ratio,
        'concentration': cv_df['concentration_index']
    }).fillna(0)
    
    risk_df['risk_score'] = (
        (1 - (risk_df['total_demographic'] / (risk_df['total_demographic'].max() + 1))) +
        (1 - (risk_df['child_ratio'] / (risk_df['child_ratio'].max() + 1))) +
        (risk_df['concentration'] / (risk_df['concentration'].max() + 1))
    )
    
    def classify_risk(x):
        if x > 2: return "🔴 HIGH RISK"
        elif x > 1: return "🟡 MEDIUM RISK"
        else: return "🟢 LOW RISK"
    
    risk_df['risk_level'] = risk_df['risk_score'].apply(classify_risk)
    risk_df['intensity_index'] = intensity_index
    
    return risk_df

def projection_analysis():
    """Project demographic trends for next cycle."""
    growth_rate = (
        (age_summary['demo_age_17_'] - age_summary['demo_age_5_17']) / 
        age_summary['demo_age_5_17'].replace(0, np.nan)
    ).replace([np.inf, -np.inf], 0).fillna(0)
    
    projection_df = pd.DataFrame({
        'current_total': district_total,
        'growth_rate': growth_rate
    })
    projection_df['projected_next_cycle'] = (
        projection_df['current_total'] * (1 + projection_df['growth_rate'].clip(-0.5, 1))
    )
    projection_df['growth_potential'] = (
        ((projection_df['projected_next_cycle'] - projection_df['current_total']) / 
         projection_df['current_total'] * 100).round(2)
    )
    
    return projection_df

def pareto_analysis():
    """Analyze Pareto distribution of enrollments."""
    sorted_vals = district_total.sort_values(ascending=False)
    cum_share = sorted_vals.cumsum() / sorted_vals.sum()
    
    print("\n📊 PARETO ANALYSIS")
    print(f"   Districts contributing to 50% enrollment: {(cum_share <= 0.5).sum()}")
    print(f"   Districts contributing to 80% enrollment: {(cum_share <= 0.8).sum()}")
    print(f"   Total districts: {len(sorted_vals)}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(cum_share)+1), cum_share.values * 100, 
             marker='o', linewidth=2, markersize=5, color='#2E86AB')
    plt.axhline(50, linestyle='--', color='red', label='50% Mark', linewidth=2)
    plt.axhline(80, linestyle='--', color='orange', label='80% Mark', linewidth=2)
    plt.xlabel("District Rank", fontsize=11, fontweight='bold')
    plt.ylabel("Cumulative Share (%)", fontsize=11, fontweight='bold')
    plt.title("Pareto Analysis - District Contribution to Total Demographic", 
              fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return cum_share
def single_district_analysis():
    """Analyze demographic patterns by pincode for a single district."""
    district_name = input("Enter District Name: ").strip().lower()
    
    district_df = df[df['district'] == district_name]
    
    if district_df.empty:
        print(f"❌ No data found for district: {district_name}")
        return
    
    # Aggregate by pincode
    agg_df = (
        district_df
        .groupby('pincode')[AGE_COLUMNS]
        .sum()
        .reset_index()
        .sort_values(by='pincode')
    )
    
    # Statistics
    total_sum = agg_df[AGE_COLUMNS].sum().sum()
    num_pincodes = len(agg_df)
    avg_per_pincode = total_sum / num_pincodes if num_pincodes > 0 else 0
    
    print(f"\n📊 {district_name.upper()} ANALYSIS")
    print(f"   Total Demographics: {int(total_sum):,}")
    print(f"   Number of Pincodes: {num_pincodes}")
    print(f"   Average per Pincode: {avg_per_pincode:,.0f}")
    
    # Visualization
    plt.figure(figsize=(15, 7))
    plt.plot(agg_df['pincode'], agg_df['demo_age_5_17'], marker='s', label='Age 5-17', linewidth=2)
    plt.plot(agg_df['pincode'], agg_df['demo_age_17_'], marker='^', label='Age 17+', linewidth=2)
    
    plt.xlabel("Pincode", fontsize=12, fontweight='bold')
    plt.ylabel("Total Demographics", fontsize=12, fontweight='bold')
    plt.title(f"{district_name.title()} - Aadhaar Demographics by Age Group (Per Pincode)", 
              fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, fontsize=9)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Uncomment to run:
# single_district_analysis()



# =========================
# ANALYSIS 2: MULTI-DISTRICT COMPARISON
# =========================
def multi_district_comparison():
    """Compare demographic metrics across 2-5 districts."""
    district_input = input("\nEnter 2 to 5 district names (comma separated): ").lower()
    district_list = [d.strip() for d in district_input.split(",")]
    
    if len(district_list) < 2 or len(district_list) > 5:
        print("❌ Error: Please enter between 2 and 5 districts.")
        return
    
    comp_df = df[df['district'].isin(district_list)]
    
    if comp_df.empty:
        print("❌ No data found for given districts.")
        return
    
    # Aggregate per district
    district_compare = (
        comp_df
        .groupby('district')[AGE_COLUMNS]
        .sum()
        .reset_index()
    )
    
    district_compare['total_demographic'] = district_compare[AGE_COLUMNS].sum(axis=1)
    district_compare = district_compare.sort_values(by='total_demographic', ascending=False)
    
    print("\n📈 MULTI-DISTRICT COMPARISON")
    print(district_compare.to_string(index=False))
    
    # Visualization 1: Total Demographic
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Line plot
    axes[0].plot(district_compare['district'], district_compare['total_demographic'], 
                 marker='o', linewidth=2, markersize=10, color='#2E86AB')
    axes[0].set_xlabel("District", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Total Demographics", fontsize=11, fontweight='bold')
    axes[0].set_title("Total Demographics by District", fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # Bar plot for age groups
    district_compare.set_index('district')[AGE_COLUMNS].plot(
        kind='bar', ax=axes[1], width=0.8, color=['#A23B72', '#F18F01', '#C73E1D']
    )
    axes[1].set_xlabel("District", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Demographics", fontsize=11, fontweight='bold')
    axes[1].set_title("Age Group-wise Distribution", fontsize=12, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend(title='Age Group', fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

# Uncomment to run:
# multi_district_comparison()

# =========================
# STATISTICAL METRICS & KPIs
# =========================

def growth_rate_analysis():
    """Calculate growth rate between demographic segments."""
    age_growth = df.groupby('district')[AGE_COLUMNS].sum()
    
    age_growth['growth_demo_ratio'] = (
        (age_growth['demo_age_17_'] - age_growth['demo_age_5_17']) / 
        age_growth['demo_age_5_17'].replace(0, np.nan)
    )
    
    age_growth = age_growth.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print("\n📊 METRIC 1: GROWTH RATE BETWEEN DEMOGRAPHIC SEGMENTS")
    print("(Top 10 Districts)")
    print(age_growth.sort_values('growth_demo_ratio', ascending=False).head(10))
    return age_growth

def demographic_intensity_index():
    """Calculate demographic pressure per pincode."""
    district_total = df.groupby('district')['total_demographic'].sum()
    pincode_count = df.groupby('district')['pincode'].nunique()
    
    intensity_index = (district_total / pincode_count).sort_values(ascending=False)
    
    print("\n📊 METRIC 2: DEMOGRAPHIC INTENSITY INDEX")
    print("(Pressure per Pincode - Top 10)")
    print(intensity_index.head(10))
    return intensity_index

def child_inclusion_ratio():
    """Calculate ratio between demographic segments."""
    child_ratio = df.groupby('district')[['demo_age_5_17', 'demo_age_17_']].sum()
    child_ratio['demographic_ratio'] = (
        child_ratio['demo_age_5_17'] / 
        child_ratio['demo_age_17_'].replace(0, np.nan)
    )
    
    child_ratio = child_ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print("\n📊 METRIC 3: DEMOGRAPHIC RATIO")
    print("(Age 5-17 to Age 17+ Ratio - Top 10)")
    print(child_ratio.sort_values('demographic_ratio', ascending=False).head(10))
    return child_ratio

def pincode_inequality_analysis():
    """Analyze intra-district inequality across pincodes."""
    district_name = input("\nEnter district name for inequality analysis: ").lower().strip()
    
    d_df = df[df['district'] == district_name]
    
    if d_df.empty:
        print(f"❌ No data found for: {district_name}")
        return
    
    pincode_sum = d_df.groupby('pincode')['total_demographic'].sum()
    
    print(f"\n📊 METRIC 4: PINCODE INEQUALITY IN {district_name.upper()}")
    print(f"   Mean: {pincode_sum.mean():,.2f}")
    print(f"   Std Dev: {pincode_sum.std():,.2f}")
    print(f"   Coeff of Variation: {(pincode_sum.std() / pincode_sum.mean()):.4f}")
    
    # Visualization
    plt.figure(figsize=(14, 6))
    pincode_sum.sort_values().plot(kind='bar', color='#2E86AB')
    plt.title(f"Pincode-wise Demographic Distribution - {district_name.title()}", 
              fontsize=13, fontweight='bold')
    plt.xlabel("Pincode", fontsize=11, fontweight='bold')
    plt.ylabel("Total Demographics", fontsize=11, fontweight='bold')
    plt.xticks(rotation=90, fontsize=8)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def concentration_index():
    """Calculate demographic concentration across pincodes."""
    pincode_level = df.groupby(['district', 'pincode'])['total_demographic'].sum().reset_index()
    
    cv_df = pincode_level.groupby('district')['total_demographic'].agg(['mean', 'std'])
    cv_df['concentration_index'] = cv_df['std'] / cv_df['mean']
    
    cv_df = cv_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print("\n📊 METRIC 5: CONCENTRATION INDEX (CV)")
    print("(Demographic Concentration - Top 10)")
    print(cv_df.sort_values('concentration_index', ascending=False).head(10))
    return cv_df

def adult_dependency_index():
    """Calculate demographic proportion by segment."""
    dependency_df = df.groupby('district')[['demo_age_17_', 'total_demographic']].sum()
    dependency_df['age_17_plus_ratio'] = (
        dependency_df['demo_age_17_'] / dependency_df['total_demographic']
    )
    
    print("\n📊 METRIC 6: AGE 17+ RATIO")
    print("(Age 17+ Demographic % - Top 10)")
    print(dependency_df.sort_values('age_17_plus_ratio', ascending=False).head(10))
    return dependency_df

def age_correlation_analysis():
    """Analyze correlation between age groups."""
    corr_df = df.groupby('district')[AGE_COLUMNS].sum()
    correlation_matrix = corr_df.corr()
    
    print("\n📊 METRIC 7: AGE GROUP CORRELATION")
    print(correlation_matrix.round(4))
    
    # Heatmap visualization
    plt.figure(figsize=(8, 6))
    im = plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    plt.xticks(range(len(AGE_COLUMNS)), AGE_COLUMNS, rotation=45)
    plt.yticks(range(len(AGE_COLUMNS)), AGE_COLUMNS)
    plt.title("Correlation Matrix - Age Groups", fontsize=12, fontweight='bold')
    plt.colorbar(im, label='Correlation')
    
    # Add correlation values to cells
    for i in range(len(AGE_COLUMNS)):
        for j in range(len(AGE_COLUMNS)):
            text = plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return correlation_matrix

def outlier_detection():
    """Identify districts with unusual demographic patterns."""
    district_totals = df.groupby('district')['total_demographic'].sum()
    z_scores = (district_totals - district_totals.mean()) / district_totals.std()
    outliers = z_scores[abs(z_scores) > 2.5]
    
    print("\n📊 METRIC 8: OUTLIER DISTRICTS")
    print(f"   (Z-score > 2.5 indicates outlier)")
    if len(outliers) > 0:
        for district, z_score in outliers.items():
            print(f"   {district.title()}: Z-score = {z_score:.3f}")
    else:
        print("   No significant outliers detected.")
    
    return outliers

def rank_stability_index():
    """Measure consistency of district ranking across age groups."""
    rank_df = df.groupby('district')[AGE_COLUMNS].sum()
    
    rank_df['rank_5_17'] = rank_df['demo_age_5_17'].rank(ascending=False)
    rank_df['rank_17_plus'] = rank_df['demo_age_17_'].rank(ascending=False)
    rank_df['rank_stability'] = rank_df[['rank_5_17', 'rank_17_plus']].std(axis=1)
    
    print("\n📊 METRIC 9: RANK STABILITY INDEX")
    print("(Lower values = more consistent ranking - Top 10)")
    print(rank_df.sort_values('rank_stability').head(10)[['rank_stability']])
    return rank_df

def run_all_10_metrics_batch():
    """
    Execute all 10 core metrics in batch mode with comprehensive output.
    Displays all metrics sequentially with proper formatting.
    """
    print("\n" + "="*70)
    print("EXECUTING ALL 10 CORE METRICS - BATCH ANALYSIS")
    print("="*70)
    
    # Pre-calculate shared data
    print("\n🔄 Pre-calculating data...")
    
    # METRIC 1: Growth Rate
    print("\n" + "-"*70)
    print("METRIC 1: GROWTH RATE BETWEEN AGE SEGMENTS")
    print("-"*70)
    age_growth = df.groupby('district')[AGE_COLUMNS].sum()
    age_growth['growth_demo_ratio'] = (
        (age_growth['demo_age_17_'] - age_growth['demo_age_5_17']) / 
        age_growth['demo_age_5_17'].replace(0, np.nan)
    )
    age_growth = age_growth.replace([np.inf, -np.inf], np.nan).fillna(0)
    print("Top 10 Districts by Growth (5-17 to 17+):")
    print(age_growth.sort_values('growth_demo_ratio', ascending=False).head(10))
    
    # METRIC 2: Intensity Index
    print("\n" + "-"*70)
    print("METRIC 2: DEMOGRAPHIC INTENSITY INDEX")
    print("-"*70)
    district_total = df.groupby('district')['total_demographic'].sum()
    pincode_count = df.groupby('district')['pincode'].nunique()
    intensity_index = (district_total / pincode_count).replace([np.inf, -np.inf], 0).fillna(0)
    print("Top 10 Districts by Intensity:")
    print(intensity_index.sort_values(ascending=False).head(10))
    
    # METRIC 3: Child Inclusion Ratio
    print("\n" + "-"*70)
    print("METRIC 3: CHILD INCLUSION RATIO")
    print("-"*70)
    child_ratio = df.groupby('district')[['demo_age_5_17', 'demo_age_17_']].sum()
    child_ratio['demographic_ratio'] = (
        child_ratio['demo_age_5_17'] / child_ratio['demo_age_17_'].replace(0, np.nan)
    )
    child_ratio = child_ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
    print("Top 10 Districts by Demographic Ratio:")
    print(child_ratio.sort_values('demographic_ratio', ascending=False).head(10)[['demographic_ratio']])
    
    # METRIC 4: Pincode Inequality (sample district)
    print("\n" + "-"*70)
    print("METRIC 4: PINCODE INEQUALITY (Sample - Top District)")
    print("-"*70)
    top_district = district_total.idxmax()
    top_dist_df = df[df['district'] == top_district]
    pincode_sum = top_dist_df.groupby('pincode')['total_demographic'].sum()
    print(f"District: {top_district.title()}")
    print(f"Mean: {pincode_sum.mean():,.2f}")
    print(f"Standard Deviation: {pincode_sum.std():,.2f}")
    print(f"Coefficient of Variation: {(pincode_sum.std() / pincode_sum.mean()):.4f}")
    
    # METRIC 5: Concentration Index
    print("\n" + "-"*70)
    print("METRIC 5: CONCENTRATION INDEX (CV)")
    print("-"*70)
    pincode_level = df.groupby(['district', 'pincode'])['total_demographic'].sum().reset_index()
    cv_df = pincode_level.groupby('district')['total_demographic'].agg(['mean', 'std'])
    cv_df['concentration_index'] = (cv_df['std'] / cv_df['mean']).replace([np.inf, -np.inf], 0).fillna(0)
    print("Top 10 Districts by Concentration:")
    print(cv_df.sort_values('concentration_index', ascending=False).head(10)[['concentration_index']])
    
    # METRIC 6: Adult Dependency Index
    print("\n" + "-"*70)
    print("METRIC 6: ADULT DEPENDENCY INDEX")
    print("-"*70)
    dependency_df = df.groupby('district')[['demo_age_17_', 'total_demographic']].sum()
    dependency_df['age_17_plus_ratio'] = (
        dependency_df['demo_age_17_'] / dependency_df['total_demographic']
    )
    print("Top 10 Districts by Age 17+ Ratio:")
    print(dependency_df.sort_values('age_17_plus_ratio', ascending=False).head(10)[['age_17_plus_ratio']])
    
    # METRIC 7: Correlation Matrix
    print("\n" + "-"*70)
    print("METRIC 7: AGE GROUP CORRELATION MATRIX")
    print("-"*70)
    corr_df = df.groupby('district')[AGE_COLUMNS].sum()
    correlation_matrix = corr_df.corr()
    print(correlation_matrix.round(4))
    
    # METRIC 8: Outlier Detection
    print("\n" + "-"*70)
    print("METRIC 8: OUTLIER DISTRICTS (Z-score > 2.5)")
    print("-"*70)
    z_scores = (district_total - district_total.mean()) / district_total.std()
    outliers = z_scores[abs(z_scores) > 2.5]
    if len(outliers) > 0:
        for district_name, z_score in outliers.items():
            print(f"   {district_name.title()}: Z-score = {z_score:.3f}")
    else:
        print("   No significant outliers detected.")
    
    # METRIC 9: Rank Stability
    print("\n" + "-"*70)
    print("METRIC 9: RANK STABILITY INDEX")
    print("-"*70)
    rank_df = df.groupby('district')[AGE_COLUMNS].sum()
    rank_df['rank_5_17'] = rank_df['demo_age_5_17'].rank(ascending=False)
    rank_df['rank_17_plus'] = rank_df['demo_age_17_'].rank(ascending=False)
    rank_df['rank_stability'] = rank_df[['rank_5_17', 'rank_17_plus']].std(axis=1)
    print("Top 10 Most Stable Districts (Lower values = more consistent):")
    print(rank_df.sort_values('rank_stability').head(10)[['rank_stability']])
    
    # METRIC 10: Composite Efficiency
    print("\n" + "-"*70)
    print("METRIC 10: COMPOSITE DISTRICT EFFICIENCY SCORE")
    print("-"*70)
    
    efficiency = pd.DataFrame()
    efficiency['total_demographic'] = district_total
    efficiency['child_ratio'] = child_ratio['demographic_ratio']
    efficiency['concentration'] = cv_df['concentration_index']
    efficiency = efficiency.fillna(0)
    
    efficiency['total_norm'] = (
        (efficiency['total_demographic'] - efficiency['total_demographic'].min()) / 
        (efficiency['total_demographic'].max() - efficiency['total_demographic'].min() + 1)
    )
    efficiency['child_norm'] = (
        (efficiency['child_ratio'] - efficiency['child_ratio'].min()) / 
        (efficiency['child_ratio'].max() - efficiency['child_ratio'].min() + 1)
    )
    efficiency['concentration_norm'] = 1 - (
        (efficiency['concentration'] - efficiency['concentration'].min()) / 
        (efficiency['concentration'].max() - efficiency['concentration'].min() + 1)
    )
    
    efficiency['efficiency_score'] = (
        (efficiency['total_norm'] + efficiency['child_norm'] + efficiency['concentration_norm']) / 3
    )
    
    print("Top 10 Districts by Efficiency Score:")
    print(efficiency.sort_values('efficiency_score', ascending=False).head(10)[['efficiency_score']])
    
    print("\n" + "="*70)
    print("✅ ALL 10 METRICS ANALYSIS COMPLETE!")
    print("="*70)
    
    return {
        'growth': age_growth,
        'intensity': intensity_index,
        'child_ratio': child_ratio,
        'concentration': cv_df,
        'dependency': dependency_df,
        'correlation': correlation_matrix,
        'outliers': outliers,
        'rank_stability': rank_df,
        'efficiency': efficiency
    }

def comprehensive_efficiency_score(age_growth, child_ratio_data, cv_df):
    """Calculate composite district efficiency score."""
    district_totals = df.groupby('district')['total_demographic'].sum()
    
    efficiency = pd.DataFrame()
    efficiency['total_demographic'] = district_totals
    efficiency['child_ratio'] = child_ratio_data['demographic_ratio']
    efficiency['concentration'] = cv_df['concentration_index']
    
    efficiency = efficiency.fillna(0)
    
    # Normalize scores (0-1)
    efficiency['total_norm'] = (
        (efficiency['total_demographic'] - efficiency['total_demographic'].min()) / 
        (efficiency['total_demographic'].max() - efficiency['total_demographic'].min() + 1)
    )
    efficiency['child_norm'] = (
        (efficiency['child_ratio'] - efficiency['child_ratio'].min()) / 
        (efficiency['child_ratio'].max() - efficiency['child_ratio'].min() + 1)
    )
    efficiency['concentration_norm'] = 1 - (
        (efficiency['concentration'] - efficiency['concentration'].min()) / 
        (efficiency['concentration'].max() - efficiency['concentration'].min() + 1)
    )
    
    # Composite score (equal weight)
    efficiency['efficiency_score'] = (
        (efficiency['total_norm'] + efficiency['child_norm'] + efficiency['concentration_norm']) / 3
    )
    
    print("\n📊 METRIC 10: COMPREHENSIVE EFFICIENCY SCORE")
    print("(Composite metric based on total demographic, child inclusion, and distribution)")
    print(efficiency.sort_values('efficiency_score', ascending=False).head(10)[['efficiency_score']])
    
    return efficiency

# =========================
# INTERACTIVE ANALYSIS MODE
# =========================

def input_based_single_district():
    """Interactive single district analysis with comparisons."""
    print("\n" + "="*70)
    print("SINGLE DISTRICT ANALYSIS MODE")
    print("="*70)
    
    district = input("Enter district name: ").strip().lower()
    
    if district not in df['district'].unique():
        print(f"❌ District '{district}' not found.")
        return
    
    d_df = df[df['district'] == district]
    
    # Age breakdown
    age_data = d_df[AGE_COLUMNS].sum()
    print(f"\n📊 {district.upper()} - AGE GROUP BREAKDOWN")
    for age_col, value in age_data.items():
        print(f"   {age_col}: {int(value):,}")
    
    plt.figure(figsize=(10, 6))
    age_data.plot(kind='bar', color=['#A23B72', '#F18F01', '#C73E1D'])
    plt.title(f"{district.title()} - Age Group Distribution", fontsize=13, fontweight='bold')
    plt.ylabel("Demographics", fontsize=11, fontweight='bold')
    plt.xlabel("Age Group", fontsize=11, fontweight='bold')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    # Pincode hotspots
    pincode_hot = d_df.groupby('pincode')['total_demographic'].sum().sort_values(ascending=False).head(20)
    
    plt.figure(figsize=(14, 7))
    pincode_hot.plot(kind='bar', color='#2E86AB')
    plt.title(f"{district.title()} - Top Pincode Hotspots", fontsize=13, fontweight='bold')
    plt.ylabel("Total Demographic", fontsize=11, fontweight='bold')
    plt.xlabel("Pincode", fontsize=11, fontweight='bold')
    plt.xticks(rotation=45, fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    # Variance metrics
    variance = d_df.groupby('pincode')['total_demographic'].sum()
    cv = (variance.std() / variance.mean()) if variance.mean() > 0 else 0
    
    print(f"\n📈 PINCODE DISTRIBUTION METRICS")
    print(f"   Variance: {variance.var():,.2f}")
    print(f"   Std Dev: {variance.std():,.2f}")
    print(f"   Coeff of Variation: {cv:.4f}")
    print(f"   Min: {variance.min():,.0f}")
    print(f"   Max: {variance.max():,.0f}")
    print(f"   Mean: {variance.mean():,.0f}")

def input_based_multi_district():
    """Interactive multi-district comparison."""
    print("\n" + "="*70)
    print("MULTI-DISTRICT COMPARISON MODE")
    print("="*70)
    
    names_input = input("Enter 2-5 districts (comma separated): ").lower()
    names = [n.strip() for n in names_input.split(',')]
    names = [n for n in names if n in df['district'].unique()]
    
    if len(names) < 2:
        print(f"❌ Need at least 2 valid districts. Found: {len(names)}")
        return
    
    if len(names) > 5:
        print(f"❌ Maximum 5 districts allowed. Got: {len(names)}")
        return
    
    comp = df[df['district'].isin(names)].groupby('district')[AGE_COLUMNS].sum()
    comp['total_demographic'] = comp.sum(axis=1)
    comp = comp.sort_values('total_demographic', ascending=False)
    
    print("\n📊 COMPARISON SUMMARY")
    print(comp.to_string())
    
    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    comp['total_demographic'].plot(kind='bar', ax=axes[0], color='#2E86AB')
    axes[0].set_title("Total Demographic Comparison", fontsize=13, fontweight='bold')
    axes[0].set_ylabel("Total Demographics", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("District", fontsize=11, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    comp[AGE_COLUMNS].plot(kind='bar', ax=axes[1], width=0.8, 
                          color=['#A23B72', '#F18F01', '#C73E1D'])
    axes[1].set_title("Age Group Distribution Comparison", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("Demographics", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("District", fontsize=11, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend(title='Age Group', fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

# =========================
# COMPREHENSIVE INPUT-BASED DISTRICT ANALYSIS
# =========================

def comprehensive_input_based_analysis():
    """
    Comprehensive multi-section analysis for a single input district.
    Includes: total enrollment comparison, pincode analysis, age distribution,
    variance metrics, top pincodes, and district comparison.
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE INPUT-BASED DISTRICT ANALYSIS")
    print("="*70)
    
    district = input("Enter district name: ").strip().lower()
    
    if district not in df['district'].unique():
        print(f"❌ District '{district}' not found.")
        return
    
    d_df = df[df['district'] == district]
    all_districts_total = df.groupby('district')['total_demographic'].sum()
    
    # =============================================
    # SECTION 1: DISTRICT OVERVIEW & COMPARISON
    # =============================================
    print(f"\n{'='*70}")
    print(f"SECTION 1: DISTRICT OVERVIEW - {district.upper()}")
    print(f"{'='*70}")
    
    district_total = d_df['total_demographic'].sum()
    rank = (all_districts_total.sort_values(ascending=False) == district_total).argmax() + 1
    percentile = (district_total / all_districts_total.max()) * 100
    
    print(f"\n📊 BASIC METRICS:")
    print(f"   Total Demographics: {int(district_total):,}")
    print(f"   Rank: {rank} out of {len(all_districts_total)}")
    print(f"   Percentile: {percentile:.2f}%")
    
    # Age breakdown
    age_data = d_df[AGE_COLUMNS].sum()
    print(f"\n📈 AGE GROUP BREAKDOWN:")
    for age_col, value in age_data.items():
        pct = (value / district_total) * 100 if district_total > 0 else 0
        print(f"   {age_col}: {int(value):,} ({pct:.2f}%)")
    
    # Visualization 1: Age Group Distribution
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    age_data.plot(kind='bar', ax=axes[0, 0], color=['#A23B72', '#F18F01', '#C73E1D'])
    axes[0, 0].set_title(f"{district.title()} - Age Group Distribution", 
                         fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel("Demographics", fontsize=10, fontweight='bold')
    axes[0, 0].set_xlabel("Age Group", fontsize=10, fontweight='bold')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # =============================================
    # SECTION 2: PINCODE ANALYSIS
    # =============================================
    print(f"\n{'='*70}")
    print(f"SECTION 2: PINCODE ANALYSIS")
    print(f"{'='*70}")
    
    pincode_stats = d_df.groupby('pincode')['total_demographic'].sum().sort_values(ascending=False)
    num_pincodes = len(pincode_stats)
    
    print(f"\n📍 PINCODE METRICS:")
    print(f"   Total Pincodes: {num_pincodes}")
    print(f"   Average per Pincode: {pincode_stats.mean():,.0f}")
    print(f"   Median per Pincode: {pincode_stats.median():,.0f}")
    print(f"   Std Dev: {pincode_stats.std():,.0f}")
    print(f"   Min: {pincode_stats.min():,.0f}")
    print(f"   Max: {pincode_stats.max():,.0f}")
    
    # Coefficient of Variation
    cv = (pincode_stats.std() / pincode_stats.mean()) if pincode_stats.mean() > 0 else 0
    print(f"   Coefficient of Variation: {cv:.4f}")
    print(f"   Variance: {pincode_stats.var():,.0f}")
    
    # Visualization 2: Top 15 Pincodes
    top_15_pincodes = pincode_stats.head(15)
    axes[0, 1].barh(range(len(top_15_pincodes)), top_15_pincodes.values, color='#2E86AB')
    axes[0, 1].set_yticks(range(len(top_15_pincodes)))
    axes[0, 1].set_yticklabels(top_15_pincodes.index, fontsize=9)
    axes[0, 1].set_xlabel("Demographics", fontsize=10, fontweight='bold')
    axes[0, 1].set_title(f"Top 15 Pincodes - {district.title()}", fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # =============================================
    # SECTION 3: VARIANCE & DISTRIBUTION ANALYSIS
    # =============================================
    print(f"\n{'='*70}")
    print(f"SECTION 3: VARIANCE & DISTRIBUTION ANALYSIS")
    print(f"{'='*70}")
    
    # Pincode concentration
    top_5_pct = (pincode_stats.head(5).sum() / pincode_stats.sum()) * 100
    top_10_pct = (pincode_stats.head(10).sum() / pincode_stats.sum()) * 100
    
    print(f"\n📊 CONCENTRATION METRICS:")
    print(f"   Top 5 Pincodes Contribution: {top_5_pct:.2f}%")
    print(f"   Top 10 Pincodes Contribution: {top_10_pct:.2f}%")
    print(f"   Inequality Index (CV): {cv:.4f}")
    
    # Visualization 3: Distribution and Percentile
    axes[1, 0].hist(pincode_stats.values, bins=20, color='#F18F01', edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(pincode_stats.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {pincode_stats.mean():.0f}')
    axes[1, 0].axvline(pincode_stats.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {pincode_stats.median():.0f}')
    axes[1, 0].set_xlabel("Demographics per Pincode", fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel("Frequency", fontsize=10, fontweight='bold')
    axes[1, 0].set_title(f"Pincode Distribution - {district.title()}", fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # =============================================
    # SECTION 4: COMPARISON WITH OTHER DISTRICTS
    # =============================================
    print(f"\n{'='*70}")
    print(f"SECTION 4: DISTRICT COMPARISON")
    print(f"{'='*70}")
    
    # Compare with top 5 districts
    top_districts = all_districts_total.nlargest(6)
    if district not in top_districts.index:
        top_districts[district] = district_total
    
    comparison_df = df[df['district'].isin(top_districts.index)].groupby('district')[AGE_COLUMNS].sum()
    comparison_df['total_demographic'] = comparison_df.sum(axis=1)
    comparison_df = comparison_df.sort_values('total_demographic', ascending=False)
    
    print(f"\n📊 COMPARISON WITH TOP 6 DISTRICTS:")
    print(comparison_df[['total_demographic']].to_string())
    
    # Visualization 4: Top Districts Comparison
    comp_total = comparison_df['total_demographic']
    colors = ['#C73E1D' if d == district else '#2E86AB' for d in comp_total.index]
    axes[1, 1].bar(range(len(comp_total)), comp_total.values, color=colors)
    axes[1, 1].set_xticks(range(len(comp_total)))
    axes[1, 1].set_xticklabels([d.title() for d in comp_total.index], rotation=45, ha='right', fontsize=9)
    axes[1, 1].set_ylabel("Total Demographics", fontsize=10, fontweight='bold')
    axes[1, 1].set_title("Comparison with Top Districts", fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # =============================================
    # SECTION 5: SUMMARY & INSIGHTS
    # =============================================
    print(f"\n{'='*70}")
    print(f"SECTION 5: KEY INSIGHTS & SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n💡 KEY FINDINGS:")
    
    if cv > 0.5:
        print(f"   🔴 High Inequality: Distribution is highly unequal across pincodes (CV: {cv:.4f})")
    elif cv > 0.3:
        print(f"   🟡 Moderate Inequality: Distribution is moderately spread (CV: {cv:.4f})")
    else:
        print(f"   🟢 Low Inequality: Distribution is relatively balanced (CV: {cv:.4f})")
    
    if top_5_pct > 50:
        print(f"   🔴 High Concentration: Top 5 pincodes control {top_5_pct:.2f}% of demographic")
    elif top_5_pct > 30:
        print(f"   🟡 Moderate Concentration: Top 5 pincodes control {top_5_pct:.2f}% of demographic")
    else:
        print(f"   🟢 Low Concentration: Demographic well distributed ({top_5_pct:.2f}%)")
    
    # Age distribution insight
    max_age_group = age_data.idxmax()
    max_age_pct = (age_data[max_age_group] / district_total) * 100
    print(f"   📊 Dominant Age Group: {max_age_group} ({max_age_pct:.2f}%)")
    
    print(f"\n✅ Analysis Complete!")

# =========================
# REPORTING FUNCTIONS
# =========================

def display_risk_dashboard(risk_df):
    """Display comprehensive risk assessment."""
    print("\n" + "="*70)
    print("RISK CLASSIFICATION DASHBOARD")
    print("="*70)
    print("\n📊 TOP 15 DISTRICTS BY RISK SCORE")
    
    top_risks = risk_df.sort_values('risk_score', ascending=False).head(15)
    for idx, (district, row) in enumerate(top_risks.iterrows(), 1):
        print(f"{idx:2d}. {district.title():30s} | {row['risk_level']:15s} | Score: {row['risk_score']:6.3f}")

def display_projection_summary(projection_df):
    """Display growth projections."""
    print("\n" + "="*70)
    print("GROWTH PROJECTION ANALYSIS")
    print("="*70)
    print("\n📈 TOP 10 DISTRICTS BY GROWTH POTENTIAL")
    
    top_growth = projection_df.sort_values('growth_potential', ascending=False).head(10)
    for idx, (district, row) in enumerate(top_growth.iterrows(), 1):
        print(f"{idx:2d}. {district.title():30s} | Current: {int(row['current_total']):>10,} | " 
              f"Projected: {int(row['projected_next_cycle']):>10,} | Growth: {row['growth_potential']:>7.2f}%")

# =========================
# MAIN MENU SYSTEM
# =========================

def main_menu():
    """Interactive main menu."""
    
    # Pre-calculate all metrics on startup
    print("\n🔄 Initializing system...")
    risk_df = calculate_risk_metrics()
    projection_df = projection_analysis()
    
    print("✅ System ready!\n")
    
    while True:
        print("\n" + "="*70)
        print("AADHAAR DEMOGRAPHIC - DISTRICT INTELLIGENCE SYSTEM")
        print("="*70)
        print("\n📋 ANALYSIS OPTIONS:")
        print("   1. 📊 Risk Classification Dashboard")
        print("   2. 🎯 Top Intervention Districts (High Risk)")
        print("   3. 📈 Growth Projection Analysis")
        print("   4. 🔍 Single District Deep Dive")
        print("   5. 📊 Multi-District Comparison")
        print("   6. 🟢 Comprehensive Input-Based Analysis (NEW!)")
        print("   7. 📉 Pareto Analysis")
        print("   8. 📊 Core Metrics Report (All 10)")
        print("   0. ❌ Exit")
        
        choice = input("\nSelect option (0-8): ").strip()
        
        if choice == "1":
            display_risk_dashboard(risk_df)
        
        elif choice == "2":
            print("\n" + "="*70)
            print("TOP DISTRICTS NEEDING URGENT INTERVENTION")
            print("="*70)
            top_10 = risk_df[risk_df['risk_level'].str.contains('HIGH|MEDIUM')].sort_values(
                'risk_score', ascending=False).head(10)
            for idx, (district, row) in enumerate(top_10.iterrows(), 1):
                print(f"{idx:2d}. {district.title():30s} | {row['risk_level']:15s} | "
                      f"Score: {row['risk_score']:6.3f} | Intensity: {row['intensity_index']:8.1f}")
        
        elif choice == "3":
            display_projection_summary(projection_df)
        
        elif choice == "4":
            input_based_single_district()
        
        elif choice == "5":
            input_based_multi_district()
        
        elif choice == "6":
            comprehensive_input_based_analysis()
        
        elif choice == "7":
            pareto_analysis()
        
        elif choice == "8":
            run_all_10_metrics_batch()
        
        elif choice == "0":
            print("\n👋 Thank you for using the system. Goodbye!")
            break
        
        else:
            print("\n❌ Invalid option. Please select 0-8.")

# =========================
# PROGRAM ENTRY POINT
# =========================

if __name__ == "__main__":
    main_menu()

