# COVID-19 Vaccination Analysis - Worldometer Data

## Overview
This notebook focuses specifically on vaccination analysis using Worldometer coronavirus data. We'll avoid redundant datetime processing and focus on vaccination metrics, geographical patterns, and key insights.

## Setup and Data Import

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Set visualization styles
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12

# Load the Worldometer coronavirus data files
try:
    daily_data = pd.read_csv("worldometer_coronavirus_daily_data.csv")
    summary_data = pd.read_csv("worldometer_coronavirus_summary_data.csv")
    
    print("Successfully loaded Worldometer coronavirus data files!")
    print(f"Daily data shape: {daily_data.shape}")
    print(f"Summary data shape: {summary_data.shape}")
    
    # Display brief summary of the datasets
    print("\nDaily data columns:", daily_data.columns.tolist())
    print("\nSummary data columns:", summary_data.columns.tolist())
    
except FileNotFoundError:
    print("Error: Could not find the data files. Please ensure the files are in the correct location.")
    # Define empty DataFrames to prevent errors later
    daily_data = pd.DataFrame()
    summary_data = pd.DataFrame()

## Data Preparation & Vaccination Metrics

# Only proceed if data was loaded successfully
if not daily_data.empty and not summary_data.empty:
    # Check for vaccination-related columns
    def find_vaccine_columns(df):
        return [col for col in df.columns if any(term in col.lower() for term in ['vaccin', 'vax', 'dose'])]

    daily_vax_cols = find_vaccine_columns(daily_data)
    summary_vax_cols = find_vaccine_columns(summary_data)

    print("Vaccination-related columns found:")
    print("Daily data:", daily_vax_cols)
    print("Summary data:", summary_vax_cols)

    # If date column exists, ensure it's in datetime format
    if 'date' in daily_data.columns:
        daily_data['date'] = pd.to_datetime(daily_data['date'])

    # Create analysis dataset by merging latest daily data with population information
    # First, check which country/location column to use
    country_cols_daily = [col for col in daily_data.columns if any(term in col.lower() for term in ['country', 'location'])]
    country_cols_summary = [col for col in summary_data.columns if any(term in col.lower() for term in ['country', 'location'])]

    country_col_daily = country_cols_daily[0] if country_cols_daily else None
    country_col_summary = country_cols_summary[0] if country_cols_summary else None

    print(f"Using '{country_col_daily}' from daily data and '{country_col_summary}' from summary data as country identifiers")

    # Create enhanced dataset with vaccination percentages
    analysis_df = None  # Initialize to avoid reference errors
    
    if country_col_daily and country_col_summary and 'date' in daily_data.columns:
        # Get latest data for each country
        latest_daily = daily_data.sort_values('date').groupby(country_col_daily).last().reset_index()
        
        # Check if summary data has population information
        pop_cols = [col for col in summary_data.columns if 'population' in col.lower()]
        if pop_cols:
            pop_col = pop_cols[0]
            print(f"Using '{pop_col}' as population data source")
            
            # Join with summary data to get population
            analysis_df = latest_daily.merge(
                summary_data[[country_col_summary, pop_col]], 
                left_on=country_col_daily, 
                right_on=country_col_summary,
                how='left'
            )
            
            # Calculate vaccination metrics if the raw numbers exist
            vax_metrics = []
            
            # Check for total vaccinations
            total_vax_cols = [col for col in analysis_df.columns if any(term in col.lower() for term in ['total_vaccin', 'total_vax', 'doses_admin'])]
            if total_vax_cols:
                total_vax_col = total_vax_cols[0]
                analysis_df['total_vaccinations_per_hundred'] = (analysis_df[total_vax_col] / analysis_df[pop_col]) * 100
                vax_metrics.append('total_vaccinations_per_hundred')
                
            # Check for people vaccinated (at least one dose)
            people_vax_cols = [col for col in analysis_df.columns if any(term in col.lower() 
                                                                        for term in ['people_vaccin', 'people_vax', 'first_dose'])]
            if people_vax_cols:
                people_vax_col = people_vax_cols[0]
                analysis_df['people_vaccinated_per_hundred'] = (analysis_df[people_vax_col] / analysis_df[pop_col]) * 100
                vax_metrics.append('people_vaccinated_per_hundred')
                
            # Check for fully vaccinated
            fully_vax_cols = [col for col in analysis_df.columns if any(term in col.lower() 
                                                                        for term in ['fully_vaccin', 'fully_vax', 'second_dose', 'complete'])]
            if fully_vax_cols:
                fully_vax_col = fully_vax_cols[0]
                analysis_df['people_fully_vaccinated_per_hundred'] = (analysis_df[fully_vax_col] / analysis_df[pop_col]) * 100
                vax_metrics.append('people_fully_vaccinated_per_hundred')
                
            print(f"Created the following vaccination metrics: {vax_metrics}")
            print(analysis_df[vax_metrics].describe())
        else:
            print("No population data found - cannot calculate vaccination percentages")
else:
    print("Data loading failed - cannot proceed with analysis")
    # Set defaults to prevent errors
    country_col_daily = None
    country_col_summary = None
    analysis_df = None

## 1. Vaccination Rate Comparison Across Countries

# Only proceed if analysis_df was created
if analysis_df is not None:
    # Select diverse countries for comparison
    countries = ['USA', 'UK', 'Israel', 'India', 'Brazil', 'South Africa', 'Canada', 'Germany', 'Japan']

    # Adjust country names based on what's in our data
    country_col = country_col_daily or country_col_summary
    if country_col:
        available_countries = daily_data[country_col].unique()
        print(f"Available countries in the data: {len(available_countries)} total")
        print(f"Sample countries: {available_countries[:10]}")
        
        # Check if our selected countries need mapping to match the dataset
        # For example, 'USA' might be stored as 'United States' in the dataset
        country_mapping = {
            'USA': ['USA', 'United States', 'US', 'U.S.', 'U.S.A.'],
            'UK': ['UK', 'United Kingdom', 'Britain', 'Great Britain'],
            # Add other mappings as needed
        }
        
        # Function to find the correct country name in our dataset
        def find_country_match(country, available_list):
            if country in available_list:
                return country
            
            if country in country_mapping:
                for alt_name in country_mapping[country]:
                    if alt_name in available_list:
                        print(f"Mapped '{country}' to '{alt_name}' in the dataset")
                        return alt_name
            
            return None  # No match found
        
        # Update our countries list with available matches
        matched_countries = []
        for country in countries:
            match = find_country_match(country, available_list=available_countries)
            if match:
                matched_countries.append(match)
            else:
                print(f"No match found for '{country}' in the dataset")
        
        countries = matched_countries
        print(f"Using these countries for analysis: {countries}")

    ### Create bar chart comparing vaccination rates

    # Check if we have the necessary data
    if any('vaccinated_per_hundred' in col for col in analysis_df.columns):
        # Find the primary vaccination metric to use
        if 'people_vaccinated_per_hundred' in analysis_df.columns:
            vax_metric = 'people_vaccinated_per_hundred'
        elif 'people_fully_vaccinated_per_hundred' in analysis_df.columns:
            vax_metric = 'people_fully_vaccinated_per_hundred'
        elif 'total_vaccinations_per_hundred' in analysis_df.columns:
            vax_metric = 'total_vaccinations_per_hundred'
        else:
            vax_metric = None
            
        if vax_metric and country_col:
            # Filter for selected countries
            country_analysis = analysis_df[analysis_df[country_col].isin(countries)]
            country_analysis = country_analysis.dropna(subset=[vax_metric])
            
            if not country_analysis.empty:
                # Sort by vaccination rate
                country_analysis = country_analysis.sort_values(vax_metric, ascending=False)
                
                # Create a horizontal bar chart
                plt.figure(figsize=(12, 8))
                sns.barplot(x=vax_metric, y=country_col, data=country_analysis, palette='viridis')
                
                metric_name = vax_metric.replace('_', ' ').title()
                plt.title(f'COVID-19 {metric_name} by Country', fontsize=18)
                plt.xlabel('Percentage of Population', fontsize=14)
                plt.ylabel('Country', fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
            else:
                print(f"No vaccination data available for the selected countries")
        else:
            print("Missing required columns for vaccination comparison")
    else:
        print("Vaccination percentage data not available")
else:
    print("Analysis dataset not created - cannot proceed with country comparison")

## 2. Comparative Pie Charts: Vaccination Coverage

# Only proceed if analysis_df was created and we have necessary columns
if analysis_df is not None and country_col and 'people_vaccinated_per_hundred' in analysis_df.columns:
    # Create a grid of pie charts to compare vaccination coverage
    # Determine grid size based on number of countries
    n_countries = len(countries)
    if n_countries > 0:  # Only create plots if we have countries to analyze
        n_cols = min(3, n_countries)
        n_rows = (n_countries + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*5))
        
        # Handle different subplot configurations
        if n_rows == 1 and n_cols == 1:
            # Single subplot case
            axes = np.array([axes])
        
        # Convert to flattened array for consistent indexing
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else np.array([axes])
        
        for i, country in enumerate(countries[:len(axes_flat)]):
            country_data = analysis_df[analysis_df[country_col] == country]
            
            if not country_data.empty and not pd.isna(country_data['people_vaccinated_per_hundred'].iloc[0]):
                vax_percent = min(country_data['people_vaccinated_per_hundred'].iloc[0], 100)  # Cap at 100%
                unvax_percent = 100 - vax_percent
                
                # Create pie chart
                axes_flat[i].pie([vax_percent, unvax_percent], 
                           labels=['Vaccinated', 'Unvaccinated'], 
                           autopct='%1.1f%%',
                           colors=['#4CAF50', '#F44336'],
                           startangle=90,
                           wedgeprops={'edgecolor': 'w', 'linewidth': 1},
                           textprops={'fontsize': 12})
                
                axes_flat[i].set_title(f'{country}', fontsize=14)
            else:
                axes_flat[i].text(0.5, 0.5, 'No vaccination data available', 
                            horizontalalignment='center', verticalalignment='center')
                axes_flat[i].set_title(f'{country}', fontsize=14)
                axes_flat[i].axis('off')
        
        # Hide any unused subplots
        for j in range(min(i+1, len(axes_flat)), len(axes_flat)):
            axes_flat[j].axis('off')
        
        plt.tight_layout()
        plt.suptitle('COVID-19 Vaccination Coverage by Country', fontsize=20, y=1.05)
        plt.show()
    else:
        print("No countries available for pie chart analysis")
else:
    print("Cannot create vaccination pie charts - required data not available")

## 3. Comparing First Dose vs. Fully Vaccinated

# Only proceed if analysis_df was created and we have both metrics
if (analysis_df is not None and country_col and
    'people_vaccinated_per_hundred' in analysis_df.columns and 
    'people_fully_vaccinated_per_hundred' in analysis_df.columns):
    
    # Filter for selected countries
    country_analysis = analysis_df[analysis_df[country_col].isin(countries)]
    country_analysis = country_analysis.dropna(subset=['people_vaccinated_per_hundred', 'people_fully_vaccinated_per_hundred'])
    
    if not country_analysis.empty:
        # Sort by first dose vaccination rate
        country_analysis = country_analysis.sort_values('people_vaccinated_per_hundred', ascending=False)
        
        # Create a grouped bar chart
        plt.figure(figsize=(14, 8))
        bar_width = 0.35
        index = np.arange(len(country_analysis))
        
        plt.bar(index, country_analysis['people_vaccinated_per_hundred'], bar_width, 
                label='At Least One Dose', color='cornflowerblue')
        plt.bar(index + bar_width, country_analysis['people_fully_vaccinated_per_hundred'], bar_width,
                label='Fully Vaccinated', color='lightseagreen')
        
        plt.title('COVID-19 Vaccination Progress: First Dose vs. Fully Vaccinated', fontsize=18)
        plt.xlabel('Country', fontsize=14)
        plt.ylabel('Percentage of Population', fontsize=14)
        plt.xticks(index + bar_width/2, country_analysis[country_col], rotation=45, ha='right')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Calculate and display the gap between first dose and fully vaccinated
        country_analysis['vaccination_gap'] = country_analysis['people_vaccinated_per_hundred'] - country_analysis['people_fully_vaccinated_per_hundred']
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='vaccination_gap', y=country_col, data=country_analysis, palette='cool')
        plt.title('Gap Between First Dose and Fully Vaccinated (Percentage Points)', fontsize=18)
        plt.xlabel('Percentage Point Difference', fontsize=14) 
        plt.ylabel('Country', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("No complete vaccination data available for the selected countries")
else:
    print("First dose and fully vaccinated data not both available")

## 4. Choropleth Map of Global Vaccination Rates

# Only proceed if analysis_df was created
if analysis_df is not None and any('vaccinated_per_hundred' in col for col in analysis_df.columns):
    # Find the primary vaccination metric to use
    if 'people_vaccinated_per_hundred' in analysis_df.columns:
        vax_metric = 'people_vaccinated_per_hundred'
        vax_title = "Population With At Least One Dose (%)"
    elif 'people_fully_vaccinated_per_hundred' in analysis_df.columns:
        vax_metric = 'people_fully_vaccinated_per_hundred'
        vax_title = "Fully Vaccinated Population (%)"
    elif 'total_vaccinations_per_hundred' in analysis_df.columns:
        vax_metric = 'total_vaccinations_per_hundred'
        vax_title = "Doses Administered per 100 People"
    else:
        vax_metric = None
        
    if vax_metric and country_col:
        # We need ISO codes for the map - check if they're in our data
        iso_cols = [col for col in analysis_df.columns if any(term in col.lower() for term in ['iso', 'code'])]
        
        if iso_cols:
            iso_col = iso_cols[0]
            print(f"Using {iso_col} for country codes in the map")
            
            # Create choropleth map
            try:
                fig = px.choropleth(
                    analysis_df,
                    locations=iso_col,
                    color=vax_metric,
                    hover_name=country_col,
                    color_continuous_scale=px.colors.sequential.Viridis,
                    title=f"COVID-19 Vaccination Rates: {vax_title}",
                    labels={vax_metric: vax_title}
                )
                fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
                fig.show()
            except Exception as e:
                print(f"Error creating choropleth map with ISO codes: {e}")
                print("Attempting alternative approach with country names...")
                try:
                    fig = px.choropleth(
                        analysis_df,
                        locations=country_col,
                        locationmode="country names",
                        color=vax_metric,
                        hover_name=country_col,
                        color_continuous_scale=px.colors.sequential.Viridis,
                        title=f"COVID-19 Vaccination Rates: {vax_title}",
                        labels={vax_metric: vax_title}
                    )
                    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
                    fig.show()
                except Exception as e:
                    print(f"Error creating choropleth map with country names: {e}")
        else:
            # Alternative approach: use country names
            print("No ISO codes found, using country names for mapping")
            try:
                fig = px.choropleth(
                    analysis_df,
                    locations=country_col,
                    locationmode="country names",
                    color=vax_metric,
                    hover_name=country_col,
                    color_continuous_scale=px.colors.sequential.Viridis,
                    title=f"COVID-19 Vaccination Rates: {vax_title}",
                    labels={vax_metric: vax_title}
                )
                fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
                fig.show()
            except Exception as e:
                print(f"Error creating choropleth map: {e}")
    else:
        print("Missing required columns for choropleth map")
else:
    print("Cannot create vaccination choropleth map - required data not available")

## 5. Regional/Continental Vaccination Analysis

# Only proceed if analysis_df was created
if analysis_df is not None:
    # Check if we have continent or region information
    region_cols = [col for col in analysis_df.columns if any(term in col.lower() for term in ['continent', 'region', 'who_region'])]

    if region_cols:
        region_col = region_cols[0]
        print(f"Using {region_col} for regional analysis")
        
        # Find the primary vaccination metric to use
        if 'people_vaccinated_per_hundred' in analysis_df.columns:
            vax_metric = 'people_vaccinated_per_hundred'
        elif 'people_fully_vaccinated_per_hundred' in analysis_df.columns:
            vax_metric = 'people_fully_vaccinated_per_hundred'
        elif 'total_vaccinations_per_hundred' in analysis_df.columns:
            vax_metric = 'total_vaccinations_per_hundred'
        else:
            vax_metric = None
            
        if vax_metric:
            # Group by region and calculate average vaccination rate
            region_vax = analysis_df.groupby(region_col)[vax_metric].mean().reset_index()
            region_vax = region_vax.dropna().sort_values(vax_metric, ascending=False)
            
            if not region_vax.empty:
                plt.figure(figsize=(12, 6))
                sns.barplot(x=region_col, y=vax_metric, data=region_vax, palette='muted')
                
                metric_name = vax_metric.replace('_', ' ').title()
                plt.title(f'Average {metric_name} by {region_col.title()}', fontsize=18)
                plt.xlabel(region_col.title(), fontsize=14)
                plt.ylabel(metric_name, fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
                
                # Create boxplots to show distribution within regions
                plt.figure(figsize=(14, 8))
                sns.boxplot(x=region_col, y=vax_metric, data=analysis_df.dropna(subset=[vax_metric]), palette='Set3')
                plt.title(f'Distribution of {metric_name} by {region_col.title()}', fontsize=18)
                plt.xlabel(region_col.title(), fontsize=14)
                plt.ylabel(metric_name, fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
            else:
                print("No valid regional vaccination data available")
        else:
            print("No vaccination metric available for regional analysis")
    else:
        print("No continent/region column found for regional analysis")
else:
    print("Analysis dataset not created - cannot proceed with regional analysis")

## 6. Vaccination Impact Analysis: Cases and Deaths

# Only proceed if analysis_df was created
if analysis_df is not None:
    # Check if we have the necessary metrics for impact analysis
    case_cols = [col for col in analysis_df.columns if any(term in col.lower() for term in ['total_cases', 'cases_per'])]
    death_cols = [col for col in analysis_df.columns if any(term in col.lower() for term in ['total_deaths', 'deaths_per'])]

    # Prefer per-capita metrics if available
    case_col = next((col for col in case_cols if 'per' in col.lower()), next(iter(case_cols), None) if case_cols else None)
    death_col = next((col for col in death_cols if 'per' in col.lower()), next(iter(death_cols), None) if death_cols else None)

    if case_col and death_col and any('vaccinated_per_hundred' in col for col in analysis_df.columns) and country_col:
        # Find the primary vaccination metric to use
        if 'people_fully_vaccinated_per_hundred' in analysis_df.columns:
            vax_metric = 'people_fully_vaccinated_per_hundred'
            vax_title = "Fully Vaccinated Population (%)"
        elif 'people_vaccinated_per_hundred' in analysis_df.columns:
            vax_metric = 'people_vaccinated_per_hundred'
            vax_title = "Population With At Least One Dose (%)"
        elif 'total_vaccinations_per_hundred' in analysis_df.columns:
            vax_metric = 'total_vaccinations_per_hundred'
            vax_title = "Doses Administered per 100 People"
        else:
            vax_metric = None
            
        if vax_metric:
            # Create scatter plots
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            
            # Vaccination vs Cases
            impact_data = analysis_df.dropna(subset=[vax_metric, case_col, death_col])
            
            if not impact_data.empty and len(countries) > 0:
                # Highlight selected countries
                selected_countries_data = impact_data[impact_data[country_col].isin(countries)]
                other_countries_data = impact_data[~impact_data[country_col].isin(countries)]
                
                # Plot other countries first (as background)
                if not other_countries_data.empty:
                    axes[0].scatter(
                        other_countries_data[vax_metric], 
                        other_countries_data[case_col],
                        alpha=0.4,
                        color='gray',
                        s=30
                    )
                
                # Then plot selected countries with labels
                for country in countries:
                    country_data = impact_data[impact_data[country_col] == country]
                    if not country_data.empty:
                        axes[0].scatter(
                            country_data[vax_metric], 
                            country_data[case_col],
                            s=100,
                            label=country
                        )
                        axes[0].text(
                            country_data[vax_metric].iloc[0] + 0.5, 
                            country_data[case_col].iloc[0],
                            country,
                            fontsize=10
                        )
                
                case_title = case_col.replace('_', ' ').title()
                axes[0].set_title(f'Vaccination Rate vs. {case_title}', fontsize=16)
                axes[0].set_xlabel(vax_title, fontsize=14)
                axes[0].set_ylabel(case_title, fontsize=14)
                axes[0].grid(True, alpha=0.3)
                
                # Vaccination vs Deaths - same approach
                if not other_countries_data.empty:
                    axes[1].scatter(
                        other_countries_data[vax_metric], 
                        other_countries_data[death_col],
                        alpha=0.4,
                        color='gray',
                        s=30
                    )
                
                for country in countries:
                    country_data = impact_data[impact_data[country_col] == country]
                    if not country_data.empty:
                        axes[1].scatter(
                            country_data[vax_metric], 
                            country_data[death_col],
                            s=100,
                            label=country
                        )
                        axes[1].text(
                            country_data[vax_metric].iloc[0] + 0.5, 
                            country_data[death_col].iloc[0],
                            country,
                            fontsize=10
                        )
                
                death_title = death_col.replace('_', ' ').title()
                axes[1].set_title(f'Vaccination Rate vs. {death_title}', fontsize=16)
                axes[1].set_xlabel(vax_title, fontsize=14)
                axes[1].set_ylabel(death_title, fontsize=14)
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
                
                # Calculate and display correlations
                corr_cases = impact_data[vax_metric].corr(impact_data[case_col])
                corr_deaths = impact_data[vax_metric].corr(impact_data[death_col])
                
                print(f"Correlation between {vax_metric} and {case_col}: {corr_cases:.3f}")
                print(f"Correlation between {vax_metric} and {death_col}: {corr_deaths:.3f}")
            else:
                print("Insufficient data for impact analysis scatter plots")
        else:
            print("No vaccination metric available for impact analysis")
    else:
        print("Cannot perform vaccination impact analysis - required metrics not available")
else:
    print("Analysis dataset not created - cannot proceed with impact analysis")

## 7. Key Insights from Vaccination Analysis

# This will be populated after running the analysis
print("# Key Insights from COVID-19 Vaccination Analysis")
print("\nBased on the analysis of Worldometer coronavirus data, here are key findings:")

# These insights will be adjusted based on the actual analysis results
insights = [
    "1. **Vaccination Rate Disparities**: There are substantial differences in vaccination coverage across countries, with some achieving high rates while others remain much lower.",
    
    "2. **First Dose vs. Full Vaccination Gap**: Many countries show a significant gap between people with at least one dose and those fully vaccinated, highlighting challenges in completing vaccination series.",
    
    "3. **Regional Patterns**: Clear regional differences exist in vaccination rates, with [specific regions] showing consistently higher rates than others.",
    
    "4. **Vaccination and Outcomes**: Countries with higher vaccination rates tend to show [specific pattern] in case and death metrics compared to those with lower rates.",
    
    "5. **Outlier Countries**: Several countries stand out in the analysis: [specific examples] demonstrate notably successful vaccination campaigns, while [others] face significant challenges."
]

# Print the insights
for insight in insights:
    print(insight)
    print()

# Limitations section
print("\n# Data Limitations")
limitations = [
    "- **Data Completeness**: Some countries may have incomplete or inconsistently reported vaccination data.",
    "- **Time Considerations**: Countries began vaccination campaigns at different times, making direct comparisons challenging.",
    "- **Different Vaccines**: Countries used different vaccines with varying dosing schedules, efficacy rates, and reporting standards.",
    "- **Contextual Factors**: Many factors beyond vaccination affect COVID-19 outcomes, including testing policies, healthcare capacity, and demographic differences.",
    "- **Reporting Standards**: Countries may define and count vaccinations differently."
]

for limitation in limitations:
    print(limitation)
