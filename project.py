import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("Electric_Vehicle_Population_Data.csv")

# Preview the data
print("\n--- Head of the Data ---")
print(df.head())

# Dataset shape and summary
print("\n--- Dataset Shape ---")
print(df.shape)
print("\n--- Column Names ---")
print(df.columns)
print("\n--- Dataset Info ---")
print(df.info())

# Check for missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Drop missing values and duplicates for clean analysis
df_clean = df.dropna().drop_duplicates()

# Count and display EVs by model year and top manufacturers
print("\n--- EV Count by Model Year ---")
print(df_clean['Model Year'].value_counts())
print("\n--- Top 10 EV Manufacturers ---")
print(df_clean['Make'].value_counts().head(10))

# --------------- INDIVIDUAL PLOTS ---------------
# 1. Number of EVs by Model Year
plt.figure(figsize=(10,5))
sns.countplot(data=df_clean, x='Model Year', order=sorted(df_clean['Model Year'].unique()))
plt.title('Number of EVs by Model Year')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Top 10 Cities with Most EVs
top_cities = df['City'].value_counts().head(10)
plt.figure(figsize=(10,6))
top_cities.plot(kind='barh', title='Top 10 Cities with Most EVs', color='skyblue')
plt.xlabel('Number of Vehicles')
plt.ylabel('City')
plt.tight_layout()
plt.show()

# 3. Electric Range Distribution
plt.figure(figsize=(10,6))
df['Electric Range'].dropna().plot(kind='hist', bins=30, title='Electric Range Distribution', color='green')
plt.xlabel('Electric Range (miles)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 4. Base MSRP Distribution
plt.figure(figsize=(10,6))
df['Base MSRP'].dropna().plot(kind='hist', bins=30, title='Base MSRP Distribution', color='purple')
plt.xlabel('Base MSRP (USD)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 5. Pie Chart: Electric Vehicle Type Distribution
type_counts = df['Electric Vehicle Type'].value_counts()
plt.figure(figsize=(7,7))
plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=140, colors=['#66b3ff','#99ff99'])
plt.title('Electric Vehicle Type Distribution')
plt.axis('equal')
plt.tight_layout()
plt.show()

# 6. Line Chart: EV Growth Over Years
ev_by_year = df['Model Year'].value_counts().sort_index()
plt.figure(figsize=(10,6))
plt.plot(ev_by_year.index, ev_by_year.values, marker='o', color='orange')
plt.title('Electric Vehicle Registrations Over the Years')
plt.xlabel('Model Year')
plt.ylabel('Number of EVs')
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Correlation Heatmap: MSRP vs Electric Range
corr_data = df[['Base MSRP', 'Electric Range']].dropna()
corr = corr_data.corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation between MSRP and Electric Range')
plt.tight_layout()
plt.show()

# 8. Top 10 EV Makes
plt.figure(figsize=(10,6))
top_makes = df_clean['Make'].value_counts().head(10)
sns.barplot(x=top_makes.values, y=top_makes.index, palette="viridis")
plt.title('Top 10 Electric Vehicle Makes')
plt.xlabel('Number of Vehicles')
plt.ylabel('Make')
plt.tight_layout()
plt.show()

# ---------------- DASHBOARD ----------------
# ---------------- DASHBOARD 1: General Insights ----------------
fig1, axs1 = plt.subplots(3, 2, figsize=(18, 16), constrained_layout=True)
fig1.suptitle('Electric Vehicle Dashboard - General Insights', fontsize=20, fontweight='bold')

# Subplot 1: Number of EVs by Model Year
sns.countplot(data=df_clean, x='Model Year', order=sorted(df_clean['Model Year'].unique()), ax=axs1[0, 0])
axs1[0, 0].set_title('EVs by Model Year', fontsize=14)
axs1[0, 0].tick_params(axis='x', rotation=45)

# Subplot 2: Top 10 Cities
top_cities.plot(kind='barh', ax=axs1[0, 1], color='skyblue')
axs1[0, 1].set_title('Top 10 Cities with Most EVs', fontsize=14)
axs1[0, 1].set_xlabel('Number of Vehicles')

# Subplot 3: Electric Range Distribution
df['Electric Range'].dropna().plot(kind='hist', bins=30, ax=axs1[1, 0], color='green')
axs1[1, 0].set_title('Electric Range Distribution', fontsize=14)
axs1[1, 0].set_xlabel('Range (miles)')

# Subplot 4: Base MSRP Distribution
df['Base MSRP'].dropna().plot(kind='hist', bins=30, ax=axs1[1, 1], color='purple')
axs1[1, 1].set_title('Base MSRP Distribution', fontsize=14)
axs1[1, 1].set_xlabel('MSRP (USD)')

# Subplot 5: EV Type Pie Chart
type_counts.plot(kind='pie', ax=axs1[2, 0], autopct='%1.1f%%', startangle=140,
                 colors=['#66b3ff', '#99ff99'], legend=False)
axs1[2, 0].set_ylabel('')
axs1[2, 0].set_title('EV Type Distribution', fontsize=14)

# Subplot 6: Blank (Placeholder for future use or note)
axs1[2, 1].axis('off')  # Leave it blank

plt.show()

# ---------------- DASHBOARD 2: Analytical Insights ----------------
fig2, axs2 = plt.subplots(2, 2, figsize=(20, 12), constrained_layout=True)
fig2.suptitle('Electric Vehicle Dashboard - Analytical Insights', fontsize=20, fontweight='bold')

# Subplot 1: Line Chart - EV Growth Over the Years
axs2[0, 0].plot(ev_by_year.index, ev_by_year.values, marker='o', color='orange')
axs2[0, 0].set_title('EV Registrations Over Years', fontsize=14)
axs2[0, 0].set_xlabel('Model Year')
axs2[0, 0].set_ylabel('Number of EVs')
axs2[0, 0].grid(True)

# Subplot 2: Heatmap - Correlation between MSRP and Range
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axs2[0, 1], fmt='.2f')
axs2[0, 1].set_title('Correlation: MSRP vs Electric Range', fontsize=14)

# Subplot 3: Top EV Makes (Horizontal Bar Chart)
sns.barplot(x=top_makes.values, y=top_makes.index, palette="viridis", ax=axs2[1, 0])
axs2[1, 0].set_title('Top 10 EV Makes', fontsize=14)
axs2[1, 0].set_xlabel('Vehicles')
axs2[1, 0].set_ylabel('Make')

# Subplot 4: Box Plot - MSRP by EV Type
sns.boxplot(data=df.dropna(), x='Electric Vehicle Type', y='Base MSRP', ax=axs2[1, 1], palette='Set2')
axs2[1, 1].set_title('Base MSRP by Electric Vehicle Type', fontsize=14)
axs2[1, 1].set_ylabel('Base MSRP (USD)')
axs2[1, 1].set_xlabel('EV Type')

plt.show()

# ---------------- SUMMARY ----------------
print("\n--- Summary and Insights ---")
print("""
- Most EVs are concentrated in cities like Seattle, Bellevue, etc.
- Electric Range generally varies between 0-350 miles with few outliers.
- Base MSRP ranges widely with peaks around $30K-$50K.
- Strong positive correlation between MSRP and Electric Range.
- BEVs dominate the dataset.
- Sharp increase in EV adoption after 2018.
""")