import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def run_pipeline():
    """Main pipeline execution"""
    
    print("=" * 60)
    print("CREDIT RISK SCORING SYSTEM")
    print("=" * 60)
    
    # Initialize configuration
    config = Config()
    
    # Step 1: Generate/load data
    print("\n1. Loading data...")
    data_gen = DataGenerator()
    df = data_gen.generate_data(n_samples=10000, seed=config.RANDOM_STATE)
    
    # Save raw data
    df.to_csv(config.RAW_DATA_PATH, index=False)
    print(f"Generated {len(df)} samples")
    print(f"Default rate: {df['default'].mean():.2%}")
    
    # Step 2: Exploratory Data Analysis
    print("\n2. Performing Exploratory Data Analysis...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Distribution of target
    df['default'].value_counts().plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of Default')
    axes[0, 0].set_xlabel('Default')
    axes[0, 0].set_ylabel('Count')
    
    # Distribution of credit score
    axes[0, 1].hist(df['credit_score'], bins=30, edgecolor='black')
    axes[0, 1].set_title('Distribution of Credit Score')
    axes[0, 1].set_xlabel('Credit Score')
    axes[0, 1].set_ylabel('Frequency')
    
    # Income vs default
    df.boxplot(column='income', by='default', ax=axes[0, 2])
    axes[0, 2].set_title('Income by Default Status')
    axes[0, 2].set_xlabel('Default')
    axes[0, 2].set_ylabel('Income')
    
    # Credit utilization vs default
    df.boxplot(column='credit_utilization', by='default', ax=axes[1, 0])
    axes[1, 0].set_title('Credit Utilization by Default Status')
    axes[1, 0].set_xlabel('Default')
    axes[1, 0].set_ylabel('Credit Utilization')
    
    # Debt to income ratio
    axes[1, 1].hist(df['debt_to_income'], bins=30, edgecolor='black')
    axes[1, 1].set_title('Distribution of Debt-to-Income Ratio')
    axes[1, 1].set_xlabel('Debt-to-Income Ratio')
    axes[1, 1].set_ylabel('Frequency')
    
    # Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', 
                center=0, ax=axes[1, 2])
    axes[1, 2].set_title('Correlation Heatmap')
    
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / 'eda_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Step 3: Split data
    print("\n3. Splitting data...")
    X = df.drop(columns=[config.TARGET])
    y = df[config.TARGET]
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=config.TEST_SIZE + config.VAL_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=config.TEST_SIZE/(config.TEST_SIZE + config.VAL_SIZE),
        random_state=config.RANDOM_STATE,
        stratify=y_temp
    )
    
    print(f"Train: {len(X_train)} samples ({len(X_train)/len(df):.1%})")
    print(f"Validation: {len(X_val)} samples ({len(X_val)/len(df):.1%})")
    print(f"Test: {len(X_test)} samples ({len(X_test)/len(df):.1%})")
    
    # Step 4: Process data
    print("\n4. Processing data...")
    data_processor = DataProcessor(config)
    pipeline = data_processor.create_preprocessor()
    
    X_train_processed, y_train, feature_names = data_processor.process_data(
        pd.concat([X_train, y_train], axis=1), train=True
    )
    X_val_processed, y_val = data_processor.process_data(
        pd.concat([X_val, y_val], axis=1), train=False
    )
    X_test_processed, y_test = data_processor.process_data(
        pd.concat([X_test, y_test], axis=1), train=False
    )
    
    print(f"Processed features: {X_train_processed.shape[1]}")
    
    # Step 5: Train models
    print("\n5. Training models...")
    trainer = ModelTrainer(config)
    best_model = trainer.train_models(
        X_train_processed, y_train,
        X_val_processed, y_val,
        feature_names
    )
    
    # Step 6: Evaluate on test set
    print("\n6. Evaluating on test set...")
    y_test_pred = best_model.predict(X_test_processed)
    y_test_proba = best_model.predict_proba(X_test_processed)[:, 1]
    
    test_metrics = trainer.calculate_metrics(y_test, y_test_pred, y_test_proba)
    print("\nTest Set Performance:")
    print(f"ROC AUC: {test_metrics['roc_auc']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    
    # Step 7: Model explanation
    print("\n7. Explaining model...")
    explainer = ModelExplainer(best_model, pipeline, feature_names)
    
    # Calculate SHAP values
    shap_values = explainer.calculate_shap_values(X_train_processed[:1000])
    
    # Plot feature importance
    explainer.plot_feature_importance(
        X_train_processed[:1000],
        y_train[:1000],
        save_path=config.REPORTS_DIR / 'feature_importance.png'
    )
    
    # Generate risk report
    risk_report = explainer.generate_risk_report(
        X_test_processed,
        y_test,
        y_test_pred,
        save_path=config.REPORTS_DIR / 'risk_report.json'
    )
    
    # Step 8: Save everything
    print("\n8. Saving artifacts...")
    data_processor.save_pipeline()
    trainer.save_model()
    
    # Save feature names
    feature_info = {
        'feature_names': feature_names,
        'feature_count': len(feature_names),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(config.MODELS_DIR / 'feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return {
        'data_processor': data_processor,
        'trainer': trainer,
        'explainer': explainer,
        'test_metrics': test_metrics
    }

if __name__ == "__main__":
    results = run_pipeline()