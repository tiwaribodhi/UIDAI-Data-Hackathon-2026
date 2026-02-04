"""
Aadhaar Demographic Analysis - Comprehensive District-Level Insights

====================================================================

This script performs in-depth analysis of Aadhaar demographic data across districts,
providing insights on infrastructure penetration, age-wise distribution, and equity metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress all glyph-related warnings from tkinter and matplotlib
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from font.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*Tight layout.*')

# Set style for consistent visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# =========================
# 1. LOAD & MERGE DATA
# =========================

BASE_PATH = r"C:\New folder\Program1\Hackathon\CSV Files\api_data_aadhar_demographic"

file1 = rf"{BASE_PATH}\api_data_aadhar_demographic_0_500000.csv"
file2 = rf"{BASE_PATH}\api_data_aadhar_demographic_500000_1000000.csv"
file3 = rf"{BASE_PATH}\api_data_aadhar_demographic_1000000_1500000.csv"
file4 = rf"{BASE_PATH}\api_data_aadhar_demographic_1500000_2000000.csv"
file5 = rf"{BASE_PATH}\api_data_aadhar_demographic_2000000_2071700.csv"

# Load CSV files
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)
df4 = pd.read_csv(file4)
df5 = pd.read_csv(file5)

# Merge all datasets
df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

# =========================
# 2. DATA CLEANING & PREPROCESSING
# =========================

# Clean district names: strip whitespace and convert to lowercase
df['district'] = df['district'].astype(str).str.strip().str.lower()

# Clean pincode: convert to string and strip whitespace
df['pincode'] = df['pincode'].astype(str).str.strip()

# Define age group columns (now 2 demographic columns)
age_cols = ['demo_age_5_17', 'demo_age_17_']

# Convert age columns to numeric and fill NaN with 0
df[age_cols] = df[age_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

# Calculate total demographic
df['total_demographic'] = df['demo_age_5_17'] + df['demo_age_17_']

print(f"Data loaded and cleaned. Total records: {len(df):,}")
print(f"Total unique districts: {df['district'].nunique()}")
print(f"Total unique pincodes: {df['pincode'].nunique()}\n")

# =========================
# UTILITY FUNCTION: Plot bars with value labels
# =========================

def plot_bar_chart(data, x_col, y_col, title, xlabel, ylabel, figsize=(14, 7)):
    """
    Create a bar chart with value labels on top of each bar.
    """
    plt.figure(figsize=figsize)
    sns.barplot(
        data=data,
        x=x_col,
        y=y_col,
        hue=x_col,
        palette="viridis",
        legend=False
    )
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=75, ha='right')

    # Add value labels
    for index, row in data.iterrows():
        plt.text(index, row[y_col], f"{int(row[y_col]):,}",
                 ha='center', va='bottom', fontsize=9)

    # Adjust margins instead of tight_layout
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.25, top=0.88)
    plt.show()

# =========================
# Q1. DISTRICTS WITH HIGHEST AADHAAR DEMOGRAPHIC INTENSITY
# =========================

district_total = (
    df.groupby('district')['total_demographic']
    .sum()
    .sort_values(ascending=False)
    .head(20)
    .reset_index()
)

plot_bar_chart(
    data=district_total,
    x_col='district',
    y_col='total_demographic',
    title="Top 20 Districts with Highest Aadhaar Demographic Intensity",
    xlabel="District",
    ylabel="Total Aadhaar Demographics"
)

# =========================
# Q2. DISTRICTS WITH CRITICAL AADHAAR DEMOGRAPHIC GAPS
# =========================

under_covered = (
    df.groupby('district')['total_demographic']
    .sum()
    .sort_values()
    .head(20)
    .reset_index()
)

plot_bar_chart(
    data=under_covered,
    x_col='district',
    y_col='total_demographic',
    title="Bottom 20 Districts with Critical Aadhaar Demographic Gaps",
    xlabel="District",
    ylabel="Total Aadhaar Demographics"
)

# =========================
# Q3. DISTRICTS WITH STRONG YOUNGER-AGE DEMOGRAPHIC INCLUSION (5–17 COHORT)
# =========================

younger_demo = (
    df.groupby('district')['demo_age_5_17']
    .sum()
    .sort_values(ascending=False)
    .head(15)
    .reset_index()
)

plot_bar_chart(
    data=younger_demo,
    x_col='district',
    y_col='demo_age_5_17',
    title="Top 15 Districts with Strong Younger-Age Aadhaar Demographic Inclusion (5–17 Years)",
    xlabel="District",
    ylabel="5–17 Age Group Demographics"
)

# =========================
# Q3B. ADULT DEMOGRAPHIC INCLUSION (17+ WORKFORCE COHORT)
# =========================

adult_demo = (
    df.groupby('district')['demo_age_17_']
    .sum()
    .sort_values(ascending=False)
    .head(15)
    .reset_index()
)

plot_bar_chart(
    data=adult_demo,
    x_col='district',
    y_col='demo_age_17_',
    title="Top 15 Districts with Strong Adult Aadhaar Demographic Inclusion (17+ Years)",
    xlabel="District",
    ylabel="17+ Age Group Demographics"
)

# =========================
# Q4. YOUNGER VS ADULT AADHAAR DEMOGRAPHIC IMBALANCE ANALYSIS
# =========================

age_summary = (
    df.groupby('district')[['demo_age_5_17', 'demo_age_17_']]
    .sum()
    .reset_index()
)

younger_mean = age_summary['demo_age_5_17'].mean()
adult_mean = age_summary['demo_age_17_'].mean()

plt.figure(figsize=(12, 9))
sns.scatterplot(
    data=age_summary,
    x='demo_age_5_17',
    y='demo_age_17_',
    s=100,
    alpha=0.6
)

plt.axvline(younger_mean, linestyle='--', color='gray', linewidth=1.5, alpha=0.7)
plt.axhline(adult_mean, linestyle='--', color='gray', linewidth=1.5, alpha=0.7)

plt.xlabel("Younger Demographic (5–17 Years)", fontsize=12, fontweight='bold')
plt.ylabel("Adult Demographic (17+ Years)", fontsize=12, fontweight='bold')
plt.title(
    "District-Level Imbalance Analysis: Younger vs Adult Aadhaar Demographic",
    fontsize=14,
    fontweight='bold',
    pad=20
)

plt.text(
    younger_mean * 0.3,
    adult_mean * 1.15,
    "High Adult\nLow Younger\n(Weak Youth Inclusion)",
    fontsize=10,
    ha='center',
    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3)
)

plt.text(
    younger_mean * 1.5,
    adult_mean * 1.15,
    "High Adult\nHigh Younger\n(Healthy District)",
    fontsize=10,
    ha='center',
    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3)
)

plt.text(
    younger_mean * 0.3,
    adult_mean * 0.2,
    "Low Adult\nLow Younger\n(Critical Zone)",
    fontsize=10,
    ha='center',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3)
)

plt.text(
    younger_mean * 1.5,
    adult_mean * 0.2,
    "Low Adult\nHigh Younger\n(Legacy Gap)",
    fontsize=10,
    ha='center',
    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3)
)

plt.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.9)
plt.show()

# =========================
# Q5. TOP 20 AND BOTTOM 20 DISTRICTS - AGE GROUP HEATMAPS
# =========================

district_total_series = df.groupby('district')['total_demographic'].sum().sort_values(ascending=False)

top40 = district_total_series.head(40).index
bottom40 = district_total_series.tail(40).index

top1_20 = top40[:20]
top21_40 = top40[20:40]

bottom1_20 = bottom40[:20]
bottom21_40 = bottom40[20:40]

top1_20_profile = (
    df[df['district'].isin(top1_20)]
    .groupby('district')[['demo_age_5_17', 'demo_age_17_']]
    .sum()
    .sort_index()
)
top21_40_profile = (
    df[df['district'].isin(top21_40)]
    .groupby('district')[['demo_age_5_17', 'demo_age_17_']]
    .sum()
    .sort_index()
)

bottom1_20_profile = (
    df[df['district'].isin(bottom1_20)]
    .groupby('district')[['demo_age_5_17', 'demo_age_17_']]
    .sum()
    .sort_index()
)
bottom21_40_profile = (
    df[df['district'].isin(bottom21_40)]
    .groupby('district')[['demo_age_5_17', 'demo_age_17_']]
    .sum()
    .sort_index()
)

fig, axes = plt.subplots(2, 2, figsize=(20, 22), constrained_layout=False)

sns.heatmap(top1_20_profile, cmap="Reds", ax=axes[0, 0], cbar=True, annot=False)
axes[0, 0].set_title("Top 1–20 Districts by Aadhaar Demographic (MAX Zone)", fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel("Age Groups", fontsize=11)
axes[0, 0].set_ylabel("District", fontsize=11)

sns.heatmap(top21_40_profile, cmap="Reds", ax=axes[0, 1], cbar=True, annot=False)
axes[0, 1].set_title("Top 21–40 Districts by Aadhaar Demographic (HIGH Zone)", fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel("Age Groups", fontsize=11)
axes[0, 1].set_ylabel("District", fontsize=11)

sns.heatmap(bottom1_20_profile, cmap="Blues", ax=axes[1, 0], cbar=True, annot=False)
axes[1, 0].set_title("Bottom 1–20 Districts by Aadhaar Demographic (MIN Zone)", fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel("Age Groups", fontsize=11)
axes[1, 0].set_ylabel("District", fontsize=11)

sns.heatmap(bottom21_40_profile, cmap="Blues", ax=axes[1, 1], cbar=True, annot=False)
axes[1, 1].set_title("Bottom 21–40 Districts by Aadhaar Demographic (LOW Zone)", fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel("Age Groups", fontsize=11)
axes[1, 1].set_ylabel("District", fontsize=11)

for ax in axes.flat:
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)

plt.subplots_adjust(left=0.06, right=0.98, bottom=0.05, top=0.95, wspace=0.25, hspace=0.25)
plt.show()

# =========================
# Q6. DISTRICTS WITH MOST UNIQUE PINCODES
# =========================

pincode_density = df.groupby('district')['pincode'].nunique().sort_values(ascending=False)
top30_pincode_districts = pincode_density.head(30).reset_index()
top30_pincode_districts.columns = ['district', 'pincode']

plot_bar_chart(
    data=top30_pincode_districts,
    x_col='district',
    y_col='pincode',
    title="Top 30 Districts by Number of Unique Pincodes (Coverage Spread)",
    xlabel="District",
    ylabel="Number of Unique Pincodes",
    figsize=(16, 10)
)

# =========================
# Q7. HIGHEST AND LOWEST PINCODE DEMOGRAPHIC
# =========================

pincode_summary = (
    df.groupby(['district', 'pincode'])['total_demographic']
    .sum()
    .reset_index()
)

pincode_sorted = pincode_summary.sort_values('total_demographic', ascending=False)

top30_pincodes = pincode_sorted.head(30).copy()
bottom30_pincodes = pincode_sorted.tail(30).copy()

top30_pincodes['label'] = top30_pincodes['district'] + " - " + top30_pincodes['pincode'].astype(str)
bottom30_pincodes['label'] = bottom30_pincodes['district'] + " - " + bottom30_pincodes['pincode'].astype(str)

plt.figure(figsize=(18, 10))
sns.barplot(
    data=top30_pincodes,
    x='label',
    y='total_demographic',
    hue='label',
    palette="YlGnBu",
    legend=False
)
plt.title("Top 30 Pincodes by Aadhaar Demographic (with District)", fontsize=14, fontweight='bold', pad=20)
plt.xlabel("District - Pincode", fontsize=12)
plt.ylabel("Total Demographics", fontsize=12)
plt.xticks(rotation=75, ha='right')

for index, row in top30_pincodes.iterrows():
    plt.text(index, row['total_demographic'], f"{int(row['total_demographic']):,}",
             ha='center', va='bottom', fontsize=8)

plt.subplots_adjust(left=0.06, right=0.98, bottom=0.3, top=0.9)
plt.show()

plt.figure(figsize=(18, 10))
sns.barplot(
    data=bottom30_pincodes,
    x='label',
    y='total_demographic',
    hue='label',
    palette="RdYlBu_r",
    legend=False
)
plt.title("Bottom 30 Pincodes by Aadhaar Demographic (with District)", fontsize=14, fontweight='bold', pad=20)
plt.xlabel("District - Pincode", fontsize=12)
plt.ylabel("Total Demographics", fontsize=12)
plt.xticks(rotation=75, ha='right')

for index, row in bottom30_pincodes.iterrows():
    plt.text(index, row['total_demographic'], f"{int(row['total_demographic']):,}",
             ha='center', va='bottom', fontsize=8)

plt.subplots_adjust(left=0.06, right=0.98, bottom=0.3, top=0.9)
plt.show()

# =========================
# Q8. INTRA-DISTRICT DEMOGRAPHIC INEQUALITY ANALYSIS
# =========================

district_pincode_demo = (
    df.groupby(['district', 'pincode'])['total_demographic']
    .sum()
    .reset_index()
)

district_inequality = (
    district_pincode_demo
    .groupby('district')['total_demographic']
    .var()
    .reset_index(name='inequality_score')
)

district_inequality = district_inequality.dropna()

most_unequal = district_inequality.sort_values('inequality_score', ascending=False).head(20)
most_balanced = district_inequality.sort_values('inequality_score', ascending=True).head(20)

plt.figure(figsize=(16, 10))
sns.barplot(
    data=most_unequal,
    x='district',
    y='inequality_score',
    hue='district',
    palette="Reds",
    legend=False
)
plt.title("Top 20 Districts with Highest Intra-District Demographic Inequality", fontsize=14, fontweight='bold', pad=20)
plt.xlabel("District", fontsize=12)
plt.ylabel("Inequality Score (Variance)", fontsize=12)
plt.xticks(rotation=75, ha='right')

for index, row in most_unequal.iterrows():
    plt.text(index, row['inequality_score'], f"{row['inequality_score']:,.0f}",
             ha='center', va='bottom', fontsize=9)

plt.subplots_adjust(left=0.06, right=0.98, bottom=0.3, top=0.9)
plt.show()

plt.figure(figsize=(16, 10))
sns.barplot(
    data=most_balanced,
    x='district',
    y='inequality_score',
    hue='district',
    palette="Greens",
    legend=False
)
plt.title("Top 20 Districts with Most Balanced Intra-District Demographic", fontsize=14, fontweight='bold', pad=20)
plt.xlabel("District", fontsize=12)
plt.ylabel("Inequality Score (Variance)", fontsize=12)
plt.xticks(rotation=75, ha='right')

for index, row in most_balanced.iterrows():
    plt.text(index, row['inequality_score'], f"{row['inequality_score']:,.0f}",
             ha='center', va='bottom', fontsize=9)

plt.subplots_adjust(left=0.06, right=0.98, bottom=0.3, top=0.9)
plt.show()

# =========================
# Q9. DEMOGRAPHIC DISTRIBUTION BY AGE GROUP - BAR CHART
# =========================

total_by_age = pd.DataFrame({
    'Age 5-17': [df['demo_age_5_17'].sum()],
    'Age 17+': [df['demo_age_17_'].sum()]
})

fig, ax = plt.subplots(figsize=(12, 7))
total_by_age.T.plot(kind='bar', ax=ax, legend=False, color=['#4ECDC4', '#45B7D1'])

plt.title("Overall Aadhaar Demographic Distribution by Age Group", fontsize=14, fontweight='bold', pad=20)
plt.xlabel("Age Group", fontsize=12)
plt.ylabel("Total Demographic", fontsize=12)
plt.xticks(rotation=0)

for i, v in enumerate(total_by_age.T[0]):
    ax.text(i, v, f"{int(v):,}", ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.9)
plt.show()

# =========================
# Q10. QUADRANT ANALYSIS – YOUNGER VS ADULT DEMOGRAPHIC (ENHANCED)
# =========================

age_summary_q = df.groupby('district')[['demo_age_5_17', 'demo_age_17_']].sum().reset_index()

younger_mean_q = age_summary_q['demo_age_5_17'].mean()
adult_mean_q = age_summary_q['demo_age_17_'].mean()

plt.figure(figsize=(12, 9))
scatter = sns.scatterplot(
    data=age_summary_q,
    x='demo_age_5_17',
    y='demo_age_17_',
    s=120,
    alpha=0.7
)

plt.axvline(younger_mean_q, linestyle='--', color='darkgray', linewidth=2, alpha=0.7, label='Mean Younger Demographic')
plt.axhline(adult_mean_q, linestyle='--', color='darkgray', linewidth=2, alpha=0.7, label='Mean Adult Demographic')

plt.xlabel("Younger Demographic (5–17 Years)", fontsize=12, fontweight='bold')
plt.ylabel("Adult Demographic (17+ Years)", fontsize=12, fontweight='bold')
plt.title("Quadrant Analysis – Younger vs Adult Aadhaar Demographic", fontsize=14, fontweight='bold', pad=20)

plt.text(
    younger_mean_q * 1.5,
    adult_mean_q * 1.15,
    "HEALTHY ZONE\n(High Younger, High Adult)",
    fontsize=10,
    ha='center',
    fontweight='bold',
    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5, edgecolor='green', linewidth=2)
)

plt.text(
    younger_mean_q * 0.3,
    adult_mean_q * 1.15,
    "WEAK YOUTH ZONE\n(Low Younger, High Adult)",
    fontsize=10,
    ha='center',
    fontweight='bold',
    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5, edgecolor='red', linewidth=2)
)

plt.text(
    younger_mean_q * 1.5,
    adult_mean_q * 0.2,
    "LEGACY GAP\n(High Younger, Low Adult)",
    fontsize=10,
    ha='center',
    fontweight='bold',
    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5, edgecolor='blue', linewidth=2)
)

plt.text(
    younger_mean_q * 0.3,
    adult_mean_q * 0.2,
    "CRITICAL ZONE\n(Low Younger, Low Adult)",
    fontsize=10,
    ha='center',
    fontweight='bold',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5, edgecolor='orange', linewidth=2)
)

plt.legend(loc='upper left', fontsize=10)
plt.subplots_adjust(left=0.08, right=0.98, bottom=0.1, top=0.9)
plt.show()

# =========================
# Q11. TOP PERFORMING VS UNDERPERFORMING DISTRICTS - METRICS COMPARISON
# =========================

top10_districts = df.groupby('district')['total_demographic'].sum().nlargest(10).index
bottom10_districts = df.groupby('district')['total_demographic'].sum().nsmallest(10).index

top10_metrics = (
    df[df['district'].isin(top10_districts)]
    .groupby('district')
    .agg({
        'total_demographic': 'sum',
        'demo_age_5_17': 'sum',
        'demo_age_17_': 'sum'
    })
    .reset_index()
)

bottom10_metrics = (
    df[df['district'].isin(bottom10_districts)]
    .groupby('district')
    .agg({
        'total_demographic': 'sum',
        'demo_age_5_17': 'sum',
        'demo_age_17_': 'sum'
    })
    .reset_index()
)

fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=False)

top10_sorted = top10_metrics.sort_values('total_demographic', ascending=True)
axes[0, 0].barh(top10_sorted['district'], top10_sorted['total_demographic'], color='#2ecc71')
axes[0, 0].set_title("Top 10 Districts - Total Demographic", fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel("Total Demographic", fontsize=11)

top10_age = top10_metrics.set_index('district')[['demo_age_5_17', 'demo_age_17_']]
top10_age.plot(kind='barh', stacked=True, ax=axes[0, 1], color=['#3498db', '#f39c12'])
axes[0, 1].set_title("Top 10 Districts - Age-wise Demographic Distribution", fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel("Demographic Count", fontsize=11)
axes[0, 1].legend(['Age 5-17', 'Age 17+'], fontsize=10)

bottom10_sorted = bottom10_metrics.sort_values('total_demographic', ascending=True)
axes[1, 0].barh(bottom10_sorted['district'], bottom10_sorted['total_demographic'], color='#e74c3c')
axes[1, 0].set_title("Bottom 10 Districts - Total Demographic", fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel("Total Demographic", fontsize=11)

bottom10_age = bottom10_metrics.set_index('district')[['demo_age_5_17', 'demo_age_17_']]
bottom10_age.plot(kind='barh', stacked=True, ax=axes[1, 1], color=['#3498db', '#f39c12'])
axes[1, 1].set_title("Bottom 10 Districts - Age-wise Demographic Distribution", fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel("Demographic Count", fontsize=11)
axes[1, 1].legend(['Age 5-17', 'Age 17+'], fontsize=10)

plt.subplots_adjust(left=0.08, right=0.98, bottom=0.07, top=0.95, wspace=0.25, hspace=0.25)
plt.show()

# =========================
# Q12. PARETO ANALYSIS – % DISTRICTS CONTRIBUTING TO 50% DEMOGRAPHIC
# =========================

district_total_demo = df.groupby('district')['total_demographic'].sum().sort_values(ascending=False)
cumulative_share = district_total_demo.cumsum() / district_total_demo.sum()
districts_50_percent = (cumulative_share <= 0.5).sum()

print("=" * 70)
print("PARETO ANALYSIS - DEMOGRAPHIC CONCENTRATION")
print("=" * 70)
print(f"Total Districts: {len(district_total_demo)}")
print(f"Districts contributing to 50% demographic: {districts_50_percent}")
print(f"Percentage of districts: {(districts_50_percent/len(district_total_demo)*100):.1f}%")
print("=" * 70 + "\n")

fig, ax1 = plt.subplots(figsize=(14, 8))

color = '#3498db'
ax1.plot(
    range(1, len(cumulative_share) + 1),
    cumulative_share.values * 100,
    marker='o',
    linewidth=2.5,
    markersize=4,
    color=color,
    label='Cumulative % Demographic'
)

ax1.axhline(50, linestyle='--', color='red', linewidth=2, alpha=0.7, label='50% Threshold')
ax1.axvline(
    districts_50_percent,
    linestyle='--',
    color='green',
    linewidth=2,
    alpha=0.7,
    label=f'{districts_50_percent} Districts = 50%'
)

ax1.fill_between(
    range(1, len(cumulative_share) + 1),
    0,
    cumulative_share.values * 100,
    alpha=0.2,
    color=color
)

ax1.set_xlabel("District Rank (High to Low Demographic)", fontsize=12, fontweight='bold')
ax1.set_ylabel("Cumulative Demographic Share (%)", fontsize=12, fontweight='bold', color=color)
ax1.set_title("Pareto Analysis – Demographic Concentration Across Districts", fontsize=14, fontweight='bold', pad=20)
ax1.set_ylim(0, 105)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11, loc='lower right')

ax1.annotate(
    f'{districts_50_percent} districts\ncontribute 50%',
    xy=(districts_50_percent, 50),
    xytext=(districts_50_percent + 20, 30),
    fontsize=11,
    fontweight='bold',
    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7, edgecolor='orange', linewidth=2),
    arrowprops=dict(arrowstyle='->', color='orange', lw=2)
)

plt.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.92)
plt.show()

# =========================
# Q13. DEMOGRAPHIC TRENDS ACROSS DISTRICTS - TOP INSIGHTS SUMMARY
# =========================

print("\n" + "=" * 70)
print("OVERALL DEMOGRAPHIC SUMMARY STATISTICS")
print("=" * 70)

print(f"\nTotal Aadhaar Demographics: {df['total_demographic'].sum():,.0f}")
print(f"Total Districts: {df['district'].nunique()}")
print(f"Total Unique Pincodes: {df['pincode'].nunique()}")

print(f"\nAge Group Breakdown:")
print(f" Age 5-17: {df['demo_age_5_17'].sum():,.0f} "
      f"({df['demo_age_5_17'].sum()/df['total_demographic'].sum()*100:.1f}%)")
print(f" Age 17+: {df['demo_age_17_'].sum():,.0f} "
      f"({df['demo_age_17_'].sum()/df['total_demographic'].sum()*100:.1f}%)")

print(f"\nDistrict Statistics:")
print(f" Highest Demographic: {district_total_demo.max():,.0f}")
print(f" Lowest Demographic: {district_total_demo.min():,.0f}")
print(f" Average Demographic: {district_total_demo.mean():,.0f}")
print(f" Median Demographic: {district_total_demo.median():,.0f}")
print(f" Std Dev: {district_total_demo.std():,.0f}")
print("=" * 70 + "\n")

print("Analysis complete. All visualizations generated successfully.")
