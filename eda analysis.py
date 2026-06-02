import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── load data ────────────────────────────────────────────────
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("=" * 55)
print("  TITANIC DATASET — EXPLORATORY DATA ANALYSIS")
print("=" * 55)

# ── 1. basic overview ────────────────────────────────────────
print("\n[1] DATASET OVERVIEW")
print(f"  Rows        : {df.shape[0]}")
print(f"  Columns     : {df.shape[1]}")
print(f"  Memory      : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
print("\n  Column Info:")
for col in df.columns:
    dtype = df[col].dtype
    nulls = df[col].isnull().sum()
    uniq  = df[col].nunique()
    print(f"    {col:<15} dtype={str(dtype):<10} nulls={nulls:<5} unique={uniq}")

# ── 2. statistical summary ───────────────────────────────────
print("\n[2] STATISTICAL SUMMARY")
print(df.describe().round(2).to_string())

# ── 3. missing values ────────────────────────────────────────
print("\n[3] MISSING VALUES")
miss = df.isnull().sum()
miss_pct = (miss / len(df) * 100).round(1)
miss_df = pd.DataFrame({'count': miss, 'percent': miss_pct})
print(miss_df[miss_df['count'] > 0].to_string())

# ── 4. target distribution ───────────────────────────────────
print("\n[4] TARGET — Survived")
vc = df['Survived'].value_counts()
print(f"  Died     : {vc[0]}  ({vc[0]/len(df)*100:.1f}%)")
print(f"  Survived : {vc[1]}  ({vc[1]/len(df)*100:.1f}%)")

# ── 5. categorical breakdowns ────────────────────────────────
print("\n[5] CATEGORICAL BREAKDOWNS")
for col in ['Pclass', 'Sex', 'Embarked']:
    print(f"\n  {col}:")
    grp = df.groupby(col)['Survived'].agg(['count', 'sum', 'mean'])
    grp.columns = ['total', 'survived', 'survival_rate']
    grp['survival_rate'] = (grp['survival_rate'] * 100).round(1)
    print(grp.to_string())

# ── 6. numerical stats by survival ──────────────────────────
print("\n[6] NUMERICAL STATS BY SURVIVAL")
for col in ['Age', 'Fare', 'SibSp', 'Parch']:
    s0 = df[df['Survived'] == 0][col].mean()
    s1 = df[df['Survived'] == 1][col].mean()
    print(f"  {col:<8}  died={s0:.2f}   survived={s1:.2f}")

# ── 7. correlation matrix ────────────────────────────────────
print("\n[7] CORRELATION WITH SURVIVAL")
num_cols = ['Survived', 'Pclass', 'Age', 'Fare', 'SibSp', 'Parch']
corr = df[num_cols].corr()['Survived'].drop('Survived').sort_values()
for feat, val in corr.items():
    bar = '█' * int(abs(val) * 20)
    sign = '+' if val > 0 else '-'
    print(f"  {feat:<10} {sign}{bar:<20} {val:.3f}")

# ── feature engineering for plots ───────────────────────────
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['AgeGroup'] = pd.cut(
    df['Age'],
    bins=[0, 12, 18, 35, 60, 100],
    labels=['Child', 'Teen', 'Adult', 'Middle-age', 'Senior']
)
df['FareBand'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Mid', 'High', 'Very High'])

# ── plots ────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 20))
fig.suptitle('Titanic EDA — Full Report', fontsize=17, fontweight='bold', y=1.005)

gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.35)

# 1. missing value heatmap
ax1 = fig.add_subplot(gs[0, 0])
miss_matrix = df.isnull().astype(int)
sns.heatmap(miss_matrix.T, ax=ax1, cbar=False, cmap='YlOrRd',
            xticklabels=False, yticklabels=True, linewidths=0)
ax1.set_title('Missing Values Map', fontweight='bold')
ax1.set_xlabel('Passengers')
ax1.tick_params(axis='y', labelsize=8)

# 2. survival count
ax2 = fig.add_subplot(gs[0, 1])
vals = df['Survived'].value_counts()
bars = ax2.bar(['Died', 'Survived'], vals.values,
               color=['#e74c3c', '#2ecc71'], width=0.45, edgecolor='none')
ax2.set_title('Survival Count', fontweight='bold')
for b, v in zip(bars, vals.values):
    ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 4,
             f'{v}\n({v/len(df)*100:.0f}%)', ha='center', fontsize=9)

# 3. class distribution
ax3 = fig.add_subplot(gs[0, 2])
pclass_counts = df['Pclass'].value_counts().sort_index()
ax3.pie(pclass_counts.values,
        labels=['1st Class', '2nd Class', '3rd Class'],
        colors=['#f1c40f', '#3498db', '#e74c3c'],
        autopct='%1.1f%%', startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
ax3.set_title('Passenger Class Split', fontweight='bold')

# 4. age distribution
ax4 = fig.add_subplot(gs[1, 0])
df['Age'].dropna().hist(ax=ax4, bins=30, color='#3498db', edgecolor='none', alpha=0.8)
ax4.axvline(df['Age'].mean(), color='red', linestyle='--', linewidth=1.5, label=f"mean={df['Age'].mean():.1f}")
ax4.axvline(df['Age'].median(), color='orange', linestyle='--', linewidth=1.5, label=f"median={df['Age'].median():.1f}")
ax4.set_title('Age Distribution', fontweight='bold')
ax4.set_xlabel('Age')
ax4.legend(fontsize=8)

# 5. fare distribution (log scale)
ax5 = fig.add_subplot(gs[1, 1])
df['Fare'].hist(ax=ax5, bins=40, color='#9b59b6', edgecolor='none', alpha=0.8)
ax5.set_title('Fare Distribution', fontweight='bold')
ax5.set_xlabel('Fare (£)')
ax5.set_ylabel('Count')

# 6. survival by sex
ax6 = fig.add_subplot(gs[1, 2])
sex_surv = df.groupby('Sex')['Survived'].mean() * 100
ax6.bar(sex_surv.index, sex_surv.values,
        color=['#3498db', '#e91e8c'], width=0.4, edgecolor='none')
ax6.set_title('Survival Rate by Sex (%)', fontweight='bold')
ax6.set_ylim(0, 100)
for i, v in enumerate(sex_surv.values):
    ax6.text(i, v + 1.5, f'{v:.1f}%', ha='center', fontsize=9)

# 7. survival by pclass + sex (grouped bar)
ax7 = fig.add_subplot(gs[2, 0])
pivot = df.groupby(['Pclass', 'Sex'])['Survived'].mean().unstack() * 100
pivot.plot(kind='bar', ax=ax7, color=['#3498db', '#e91e8c'],
           edgecolor='none', width=0.6)
ax7.set_title('Survival Rate by Class & Sex', fontweight='bold')
ax7.set_xlabel('Passenger Class')
ax7.set_ylabel('Survival Rate (%)')
ax7.set_xticklabels(['1st', '2nd', '3rd'], rotation=0)
ax7.set_ylim(0, 110)
ax7.legend(['Female', 'Male'], fontsize=8)

# 8. age vs fare scatter
ax8 = fig.add_subplot(gs[2, 1])
scatter_died = df[df['Survived'] == 0]
scatter_surv = df[df['Survived'] == 1]
ax8.scatter(scatter_died['Age'], scatter_died['Fare'],
            alpha=0.35, s=15, color='#e74c3c', label='Died')
ax8.scatter(scatter_surv['Age'], scatter_surv['Fare'],
            alpha=0.45, s=15, color='#2ecc71', label='Survived')
ax8.set_title('Age vs Fare (by Survival)', fontweight='bold')
ax8.set_xlabel('Age')
ax8.set_ylabel('Fare (£)')
ax8.legend(fontsize=8)

# 9. survival by age group
ax9 = fig.add_subplot(gs[2, 2])
ag = df.groupby('AgeGroup', observed=True)['Survived'].mean() * 100
ax9.bar(ag.index.astype(str), ag.values,
        color=sns.color_palette('Set2', len(ag)), edgecolor='none', width=0.6)
ax9.set_title('Survival Rate by Age Group (%)', fontweight='bold')
ax9.set_ylim(0, 100)
ax9.tick_params(axis='x', rotation=20)

# 10. family size vs survival
ax10 = fig.add_subplot(gs[3, 0])
fs = df.groupby('FamilySize')['Survived'].mean() * 100
ax10.plot(fs.index, fs.values, 'o-', color='#e67e22', linewidth=2, markersize=7)
ax10.fill_between(fs.index, fs.values, alpha=0.15, color='#e67e22')
ax10.set_title('Survival Rate by Family Size', fontweight='bold')
ax10.set_xlabel('Family Size')
ax10.set_ylabel('Survival Rate (%)')

# 11. fare band vs survival
ax11 = fig.add_subplot(gs[3, 1])
fb = df.groupby('FareBand', observed=True)['Survived'].mean() * 100
ax11.bar(fb.index.astype(str), fb.values,
         color=['#e74c3c', '#f39c12', '#3498db', '#2ecc71'], edgecolor='none', width=0.5)
ax11.set_title('Survival Rate by Fare Band (%)', fontweight='bold')
ax11.set_ylim(0, 100)

# 12. correlation heatmap
ax12 = fig.add_subplot(gs[3, 2])
corr_matrix = df[['Survived', 'Pclass', 'Age', 'Fare',
                   'SibSp', 'Parch', 'FamilySize', 'IsAlone']].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, ax=ax12, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, linewidths=0.4,
            annot_kws={'size': 7}, cbar_kws={'shrink': 0.8})
ax12.set_title('Correlation Heatmap', fontweight='bold')
ax12.tick_params(labelsize=7)

plt.savefig('eda_report.png', dpi=150, bbox_inches='tight')
print("\nsaved → eda_report.png")
plt.show()

# ── final insight summary ────────────────────────────────────
print("\n" + "=" * 55)
print("  KEY INSIGHTS")
print("=" * 55)
print(f"  Survival rate overall     : {df['Survived'].mean()*100:.1f}%")
print(f"  Female survival rate      : {df[df['Sex']=='female']['Survived'].mean()*100:.1f}%")
print(f"  Male survival rate        : {df[df['Sex']=='male']['Survived'].mean()*100:.1f}%")
print(f"  1st class survival        : {df[df['Pclass']==1]['Survived'].mean()*100:.1f}%")
print(f"  3rd class survival        : {df[df['Pclass']==3]['Survived'].mean()*100:.1f}%")
print(f"  Children (<=12) survival  : {df[df['Age']<=12]['Survived'].mean()*100:.1f}%")
print(f"  Travelling alone survival : {df[df['IsAlone']==1]['Survived'].mean()*100:.1f}%")
print(f"  With family survival      : {df[df['IsAlone']==0]['Survived'].mean()*100:.1f}%")
print(f"  Avg fare (survived)       : £{df[df['Survived']==1]['Fare'].mean():.2f}")
print(f"  Avg fare (died)           : £{df[df['Survived']==0]['Fare'].mean():.2f}")
print(f"  Median age overall        : {df['Age'].median():.1f}")
