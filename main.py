
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import pandas as pd
import numpy as np
import matplotlib
import warnings
import math
import traceback
import base64
from io import BytesIO
import json
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import re

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

warnings.filterwarnings('ignore')
matplotlib.use('Agg')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'bank-finance-analytics-2024'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'txt', 'json', 'xlsx', 'xls'}
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

reference_data = None
prediction_model = None
model_type = None
model_trained = False
column_info = None
time_series_info = None
scaler = None
last_prediction_date = None
last_prediction_value = None
prediction_context = {}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def detect_data_type(df, target_column=None):
    date_columns = []
    for col in df.columns:
        try:
            if df[col].dtype == 'object':
                sample = df[col].dropna().head(10).astype(str)
                if sample.str.contains(r'\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4}', regex=True).any():
                    date_columns.append(col)
        except:
            pass

    for col in df.columns:
        col_lower = col.lower()
        if any(word in col_lower for word in ['date', 'time', 'month', 'year', 'day', 'timestamp']):
            if col not in date_columns:
                date_columns.append(col)

    if date_columns:
        for date_col in date_columns:
            try:
                test_df = df.copy()
                test_df[date_col] = pd.to_datetime(test_df[date_col], errors='coerce')
                valid_dates = test_df[date_col].notna().sum()
                if valid_dates > len(df) * 0.5:
                    return 'timeseries', date_col
            except:
                continue
        if date_columns:
            return 'timeseries', date_columns[0]

    if target_column is None or target_column not in df.columns:
        return 'regression', None

    if df[target_column].dtype == 'object':
        df[target_column] = df[target_column].astype(str).str.replace(',', '')

    try:
        df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
    except:
        pass

    unique_values = df[target_column].nunique()
    if unique_values == 2:
        value_set = set(df[target_column].dropna().astype(str).str.lower())
        binary_sets = [
            {'0', '1'}, {'pass', 'fail'}, {'yes', 'no'},
            {'true', 'false'}, {'success', 'failure'}
        ]
        for binary_set in binary_sets:
            if value_set.issubset(binary_set):
                return 'classification', None
    elif unique_values <= 10 and df[target_column].dtype == 'object':
        return 'classification', None

    return 'regression', None


def clean_numeric_column(series):
    try:
        if series.dtype == 'object':
            cleaned = series.astype(str).str.replace(',', '').str.replace('$', '').str.replace('%', '')
            return pd.to_numeric(cleaned, errors='coerce')
        return pd.to_numeric(series, errors='coerce')
    except:
        return series


def create_visualization(df, feature_col, target_col, predictions=None, model_type='regression',
                         future_predictions=None, future_dates=None, clusters=None):
    plt.figure(figsize=(14, 8))
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'

    df_plot = df.copy()
    df_plot[target_col] = clean_numeric_column(df_plot[target_col])

    if model_type == 'timeseries':
        try:
            if feature_col and feature_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[feature_col]):
                dates = df[feature_col]
                dates_valid = dates.notna()
                if dates_valid.sum() > 0:
                    x_vals = dates[dates_valid].values
                    y_vals = df_plot[target_col].values[dates_valid]

                    plt.plot(x_vals, y_vals, 'b-', linewidth=2.5, label='Historical Data',
                             alpha=0.8, marker='o', markersize=5)

                    if predictions is not None and len(predictions) > 0:
                        if len(predictions) == len(x_vals):
                            plt.plot(x_vals, predictions, 'r--', linewidth=2, label='Model Predictions', alpha=0.8)

                    if future_predictions is not None and future_dates is not None and len(future_predictions) > 0:
                        plt.plot(future_dates, future_predictions, 'g--', linewidth=2.5,
                                 label=f'Future Predictions ({len(future_predictions)} periods)', alpha=0.9, marker='s',
                                 markersize=6)
                        plt.scatter(future_dates, future_predictions, color='green', s=100, zorder=5,
                                    edgecolors='black', linewidth=1.5)

                    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m/%d/%Y'))
                    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=3))
                    plt.gcf().autofmt_xdate()
                    plt.xlabel('Date', fontsize=12, fontweight='bold')
                else:
                    raise ValueError("No valid dates")
            else:
                raise ValueError("No date column")
        except:
            x_vals = np.arange(len(df_plot))
            y_vals = df_plot[target_col].values

            plt.plot(x_vals, y_vals, 'b-', linewidth=2.5, label='Historical Data',
                     alpha=0.8, marker='o', markersize=5)

            if predictions is not None and len(predictions) > 0:
                if len(predictions) == len(x_vals):
                    plt.plot(x_vals, predictions, 'r--', linewidth=2, label='Model Predictions', alpha=0.8)

            if future_predictions is not None and len(future_predictions) > 0:
                future_idx = np.arange(len(x_vals), len(x_vals) + len(future_predictions))
                plt.plot(future_idx, future_predictions, 'g--', linewidth=2.5,
                         label=f'Future Predictions ({len(future_predictions)} periods)', alpha=0.9, marker='s',
                         markersize=6)
                plt.scatter(future_idx, future_predictions, color='green', s=100, zorder=5,
                            edgecolors='black', linewidth=1.5)

            plt.xlabel('Time Period', fontsize=12, fontweight='bold')

        plt.ylabel(target_col, fontsize=12, fontweight='bold')
        plt.title(f'Time Series Analysis & Predictions for {target_col}', fontsize=16, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.2, linestyle='--')
        plt.legend(framealpha=0.9, loc='best')
        plt.tight_layout()

    elif model_type == 'classification':
        if feature_col in df.columns:
            X = clean_numeric_column(df[feature_col]).values
        else:
            X = np.arange(len(df))

        y = df[target_col].values
        if y.dtype == 'object':
            le = LabelEncoder()
            y_numeric = le.fit_transform(y)
        else:
            y_numeric = y

        plt.scatter(X, y_numeric, alpha=0.6, s=80, label='Training Data', c='#4a6fa5', edgecolors='white',
                    linewidth=0.8)

        if predictions is not None and len(predictions) > 0:
            sorted_idx = np.argsort(X)
            X_sorted = X[sorted_idx]
            pred_sorted = predictions[sorted_idx]
            plt.plot(X_sorted, pred_sorted, 'r-', linewidth=2.5, label='Decision Boundary')

        plt.xlabel(feature_col if feature_col else 'Index', fontsize=12, fontweight='bold')
        plt.ylabel(target_col, fontsize=12, fontweight='bold')
        plt.title(f'Classification Model for {target_col}', fontsize=16, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.2, linestyle='--')
        plt.legend(framealpha=0.9, loc='best')
        plt.tight_layout()

    elif model_type == 'clustering':
        if feature_col in df.columns:
            X = clean_numeric_column(df[feature_col]).values
        else:
            X = np.arange(len(df))

        y = df[target_col].values if target_col in df.columns else np.zeros(len(df))

        if clusters is not None:
            plt.scatter(X, y, c=clusters, cmap='viridis', s=100, alpha=0.7, edgecolors='white', linewidth=1)
            plt.colorbar(label='Cluster')
            plt.title(f'K-Means Clustering Results', fontsize=16, fontweight='bold', pad=20)
        else:
            plt.scatter(X, y, alpha=0.6, s=80, label='Data Points', c='#4a6fa5', edgecolors='white', linewidth=0.8)
            plt.title(f'Data Distribution for Clustering', fontsize=16, fontweight='bold', pad=20)

        plt.xlabel(feature_col if feature_col else 'Index', fontsize=12, fontweight='bold')
        plt.ylabel(target_col if target_col else 'Value', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.2, linestyle='--')
        plt.tight_layout()

    else:
        if feature_col in df.columns:
            X = clean_numeric_column(df[feature_col]).values
        else:
            X = np.arange(len(df))

        y = df_plot[target_col].values
        plt.scatter(X, y, alpha=0.6, s=80, label='Training Data', c='#4a6fa5', edgecolors='white', linewidth=0.8)

        if predictions is not None and len(predictions) > 0:
            sorted_idx = np.argsort(X)
            X_sorted = X[sorted_idx]
            pred_sorted = predictions[sorted_idx]
            plt.plot(X_sorted, pred_sorted, 'r-', linewidth=2.5, label='Regression Line')

        plt.xlabel(feature_col if feature_col else 'Index', fontsize=12, fontweight='bold')
        plt.ylabel(target_col, fontsize=12, fontweight='bold')
        plt.title(f'Regression Model for {target_col}', fontsize=16, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.2, linestyle='--')
        plt.legend(framealpha=0.9, loc='best')
        plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def train_time_series_model(series, dates=None, model_name='ARIMA', p=1, d=1, q=1):
    if len(series) < 5:
        trend = calculate_monthly_trend(series, dates)
        return {
            'model_type': 'trend_simple_average',
            'average': float(np.mean(series)),
            'last_values': [float(x) for x in series[-min(5, len(series)):]],
            'last_date': convert_to_json_serializable(dates[-1]) if dates is not None and len(dates) > 0 else None,
            'trend': float(trend),
            'last_value': float(series[-1]) if len(series) > 0 else 0.0,
            'initial_value': float(series[0]) if len(series) > 0 else 0.0,
            'series_length': len(series),
            'avg_value': float(np.mean(series))
        }

    try:
        if model_name == 'AR':
            model = AutoReg(series, lags=p)
            model_fit = model.fit()
            model_info = {
                'model_type': 'AR',
                'model': model_fit,
                'order': (p, 0, 0),
                'lags': p,
                'aic': float(model_fit.aic),
                'params': model_fit.params.tolist()
            }

        elif model_name == 'MA':
            model = ARIMA(series, order=(0, d, q))
            model_fit = model.fit()
            model_info = {
                'model_type': 'MA',
                'model': model_fit,
                'order': (0, d, q),
                'aic': float(model_fit.aic),
                'params': model_fit.params.tolist()
            }

        elif model_name == 'ARMA':
            model = ARIMA(series, order=(p, 0, q))
            model_fit = model.fit()
            model_info = {
                'model_type': 'ARMA',
                'model': model_fit,
                'order': (p, 0, q),
                'aic': float(model_fit.aic),
                'params': model_fit.params.tolist()
            }

        elif model_name == 'ARIMA':
            model = ARIMA(series, order=(p, d, q))
            model_fit = model.fit()
            model_info = {
                'model_type': 'ARIMA',
                'model': model_fit,
                'order': (p, d, q),
                'aic': float(model_fit.aic),
                'params': model_fit.params.tolist()
            }

        else:
            model = ARIMA(series, order=(p, d, q))
            model_fit = model.fit()
            model_info = {
                'model_type': 'ARIMA',
                'model': model_fit,
                'order': (p, d, q),
                'aic': float(model_fit.aic),
                'params': model_fit.params.tolist()
            }

        trend = calculate_monthly_trend(series, dates)
        model_info.update({
            'last_date': convert_to_json_serializable(dates[-1]) if dates is not None and len(dates) > 0 else None,
            'trend': float(trend),
            'last_value': float(series[-1]) if len(series) > 0 else 0.0,
            'initial_value': float(series[0]) if len(series) > 0 else 0.0,
            'series_length': len(series),
            'avg_value': float(np.mean(series))
        })

        return model_info

    except Exception as e:
        print(f"Error in time series training ({model_name}): {e}")
        trend = calculate_monthly_trend(series, dates)
        return {
            'model_type': 'trend_simple_average',
            'average': float(np.mean(series)),
            'last_values': [float(x) for x in series[-min(5, len(series)):]],
            'last_date': convert_to_json_serializable(dates[-1]) if dates is not None and len(dates) > 0 else None,
            'trend': float(trend),
            'last_value': float(series[-1]) if len(series) > 0 else 0.0,
            'initial_value': float(series[0]) if len(series) > 0 else 0.0,
            'series_length': len(series),
            'avg_value': float(np.mean(series))
        }


def calculate_monthly_trend(series, dates=None):
    if len(series) < 2:
        return 0

    try:
        if dates is not None and len(dates) == len(series):
            months = []
            start_date = dates[0]
            for d in dates:
                months_diff = (d.year - start_date.year) * 12 + (d.month - start_date.month)
                months.append(months_diff)
            x = np.array(months)
        else:
            x = np.arange(len(series))

        y = np.array(series)
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        slope = model.params[1]
        return float(slope)

    except Exception as e:
        print(f"Error calculating trend: {e}")
        if len(series) >= 2:
            return float((series[-1] - series[0]) / max(1, (len(series) - 1)))
        return 0.0


def convert_to_json_serializable(obj):

    if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)

    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)

    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d %H:%M:%S')

    elif isinstance(obj, (np.datetime64, datetime)):
        if isinstance(obj, np.datetime64):
            return pd.Timestamp(obj).strftime('%Y-%m-%d %H:%M:%S')
        else:
            return obj.strftime('%Y-%m-%d %H:%M:%S')

    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]

    elif isinstance(obj, dict):
        return {
            key: convert_to_json_serializable(value)
            for key, value in obj.items()
        }

    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass

    return obj


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload_data', methods=['POST'])
def upload_data():
    global reference_data, prediction_model, model_type, model_trained, column_info, time_series_info, scaler, prediction_context

    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})

        if not allowed_file(file.filename):
            return jsonify(
                {'success': False, 'error': f'File type not allowed. Allowed: {app.config["ALLOWED_EXTENSIONS"]}'})

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(filepath)
            elif filename.endswith('.json'):
                df = pd.read_json(filepath)
            else:
                return jsonify({'success': False, 'error': 'Unsupported file format'})
        except Exception as e:
            return jsonify({'success': False, 'error': f'Error reading file: {str(e)}'})

        if len(df.columns) < 1:
            return jsonify({'success': False, 'error': 'File must have at least 1 column'})

        date_columns = []
        for col in df.columns:
            try:
                if df[col].dtype == 'object':
                    sample = df[col].dropna().head(5).astype(str)
                    patterns = [
                        r'\d{1,2}/\d{1,2}/\d{4}',
                        r'\d{4}-\d{2}-\d{2}',
                        r'\d{2}-\d{2}-\d{4}',
                        r'\d{1,2}-\d{1,2}-\d{2}',
                        r'\d{1,2}/\d{1,2}/\d{2}'
                    ]
                    for pattern in patterns:
                        if sample.str.contains(pattern).any():
                            date_columns.append(col)
                            break
            except:
                pass

        for col in df.columns:
            col_lower = col.lower()
            if any(word in col_lower for word in ['date', 'time', 'month', 'year', 'day', 'timestamp']):
                if col not in date_columns:
                    date_columns.append(col)

        for date_col in date_columns:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            except:
                pass

        reference_data = {
            'df': df,
            'filepath': filepath,
            'filename': filename,
            'columns': df.columns.tolist(),
            'date_columns': date_columns
        }

        prediction_model = None
        model_type = None
        model_trained = False
        column_info = None
        time_series_info = None
        scaler = None
        prediction_context = {}

        preview_df = df.head(10).copy()
        for col in preview_df.columns:
            if pd.api.types.is_numeric_dtype(preview_df[col]):
                preview_df[col] = preview_df[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "NaN")

        preview_html = preview_df.to_html(
            classes='table table-striped table-bordered',
            index=False,
            na_rep='NaN'
        )

        numeric_cols = []
        for col in df.columns:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_cols.append(col)
            except:
                pass

        return jsonify({
            'success': True,
            'filename': filename,
            'columns': df.columns.tolist(),
            'numeric_columns': numeric_cols,
            'date_columns': date_columns,
            'preview': preview_html,
            'shape': {'rows': len(df), 'cols': len(df.columns)},
            'message': f'Successfully loaded {len(df)} rows with {len(df.columns)} columns'
        })

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in upload_data: {str(e)}")
        print(f"Traceback: {error_details}")
        return jsonify({'success': False, 'error': f'Error: {str(e)}'})


@app.route('/setup_model', methods=['POST'])
def setup_model():
    global reference_data, model_type, column_info, time_series_info, prediction_context

    try:
        if reference_data is None:
            return jsonify({'success': False, 'error': 'No data uploaded. Please upload data first.'})

        data = request.get_json()

        raw_feature = data.get('feature_column') or data.get('featureColumn')
        raw_target = data.get('target_column') or data.get('targetColumn')
        selected_model_type = data.get('model_type',
                                       'auto')

        df = reference_data['df']
        date_columns = reference_data['date_columns']

        if selected_model_type != 'clustering' and not raw_target:
            return jsonify({'success': False, 'error': 'Please select a target column to predict'})

        if raw_target and raw_target not in df.columns:
            return jsonify({'success': False, 'error': f'Target column "{raw_target}" not found in data'})

        feature_col = raw_feature
        target_col = raw_target

        if selected_model_type == 'auto':
            if feature_col and feature_col in date_columns:
                model_type = 'timeseries'
            elif not feature_col and target_col not in date_columns and date_columns:
                feature_col = date_columns[0]
                model_type = 'timeseries'
            else:
                detected_type, _ = detect_data_type(df, target_col)
                model_type = detected_type
        else:
            model_type = selected_model_type

        column_info = {
            'feature': feature_col,
            'target': target_col,
            'model_type': model_type,
            'is_timeseries': model_type == 'timeseries'
        }

        prediction_context = {
            'target_name': target_col,
            'target_description': 'value',
            'feature_name': feature_col if feature_col else 'Index',
            'model_type': model_type,
            'data_source': reference_data['filename'],
            'is_timeseries': model_type == 'timeseries'
        }

        if model_type == 'timeseries':
            time_series_info = {
                'date_column': feature_col,
                'target_column': target_col,
                'frequency': 'M'
            }
            problem_desc = "Time Series Forecasting (Monthly)"
        elif model_type == 'classification':
            problem_desc = f"Classification ({df[target_col].nunique()} classes)"
        elif model_type == 'clustering':
            problem_desc = "Clustering Analysis"
        else:
            problem_desc = "Regression Analysis"

        if model_type != 'clustering':
            target_stats = {
                'unique_values': int(df[target_col].nunique()),
                'data_type': str(df[target_col].dtype),
                'missing': int(df[target_col].isna().sum())
            }

            if model_type != 'classification':
                try:
                    clean_target = clean_numeric_column(df[target_col])
                    if clean_target.notna().sum() > 0:
                        target_stats.update({
                            'mean': f"{clean_target.mean():.2f}",
                            'std': f"{clean_target.std():.2f}",
                            'min': f"{clean_target.min():.2f}",
                            'max': f"{clean_target.max():.2f}",
                            'median': f"{clean_target.median():.2f}",
                            'units': 'units'
                        })
                    else:
                        target_stats.update({
                            'mean': 'N/A',
                            'std': 'N/A',
                            'min': 'N/A',
                            'max': 'N/A',
                            'median': 'N/A',
                            'units': 'unknown'
                        })
                except:
                    target_stats.update({
                        'mean': 'N/A',
                        'std': 'N/A',
                        'min': 'N/A',
                        'max': 'N/A',
                        'median': 'N/A',
                        'units': 'unknown'
                    })
        else:
            target_stats = {}

        return jsonify({
            'success': True,
            'feature_column': feature_col,
            'target_column': target_col,
            'model_type': model_type,
            'problem_description': problem_desc,
            'target_stats': target_stats,
            'prediction_context': prediction_context,
            'message': f'Model configured as {problem_desc}. Predicting: {target_col}'
        })

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in setup_model: {str(e)}")
        print(f"Traceback: {error_details}")
        return jsonify({'success': False, 'error': f'Error: {str(e)}'})


@app.route('/train_model', methods=['POST'])
def train_model():
    global prediction_model, model_trained, reference_data, model_type, column_info, time_series_info, scaler, prediction_context

    try:
        if reference_data is None or column_info is None:
            return jsonify({'success': False, 'error': 'Please setup model first by selecting columns'})

        data = request.get_json()
        model_params = data.get('model_params', {})

        df = reference_data['df'].copy()
        feature_col = column_info['feature']
        target_col = column_info['target']

        if model_type != 'clustering':
            df[target_col] = clean_numeric_column(df[target_col])
            df = df.dropna(subset=[target_col])

        if len(df) < 5:
            return jsonify({'success': False, 'error': 'Not enough data for training. Need at least 5 rows.'})

        if model_type == 'timeseries':
            try:
                ts_data = df.copy()
                if feature_col and feature_col in df.columns:
                    ts_data[feature_col] = pd.to_datetime(ts_data[feature_col], errors='coerce')
                    ts_data = ts_data.sort_values(feature_col, ascending=True)
                    dates = ts_data[feature_col].values
                else:
                    dates = None

                series = ts_data[target_col].values

                ts_model_name = model_params.get('ts_model', 'ARIMA')
                p = int(model_params.get('p', 1))
                d = int(model_params.get('d', 1))
                q = int(model_params.get('q', 1))

                prediction_model = train_time_series_model(series, dates, ts_model_name, p, d, q)

                last_date_str = None
                if dates is not None and len(dates) > 0:
                    last_date = dates[-1]
                    last_date_str = convert_to_json_serializable(last_date)
                elif 'last_date' in prediction_model:
                    last_date_str = prediction_model['last_date']

                prediction_context['last_date'] = last_date_str

                trend = prediction_model.get('trend', 0)
                last_value = series[-1] if len(series) > 0 else 0

                model_name = f"Time Series Model ({ts_model_name})"

                if len(series) > 5:
                    if ts_model_name in ['AR', 'ARIMA', 'ARMA', 'MA']:
                        try:
                            y_pred = prediction_model['model'].predict()
                            if len(y_pred) > len(series):
                                y_pred = y_pred[:len(series)]
                            elif len(y_pred) < len(series):
                                y_pred = np.concatenate([[y_pred[0]] * (len(series) - len(y_pred)), y_pred])
                        except:
                            y_pred = series.copy()
                    else:
                        y_pred = series.copy()

                    mse = mean_squared_error(series, y_pred)
                    mae = mean_absolute_error(series, y_pred)
                    r2 = r2_score(series, y_pred)
                    metrics = {
                        'mse': f"{mse:.4f}",
                        'rmse': f"{np.sqrt(mse):.4f}",
                        'mae': f"{mae:.4f}",
                        'r2_score': f"{r2:.4f}",
                        'samples': len(series),
                        'aic': f"{prediction_model.get('aic', 'N/A')}"
                    }
                else:
                    metrics = {
                        'samples': len(series),
                        'note': 'Time series model trained successfully'
                    }

                prediction_context.update({
                    'model_name': model_name,
                    'training_samples': len(series),
                    'data_range': f"{len(df)} time periods",
                    'prediction_units': 'units',
                    'last_date': last_date_str,
                    'trend': float(trend),
                    'last_value': float(last_value)
                })

            except Exception as e:
                error_details = traceback.format_exc()
                print(f"Error preparing time series data: {str(e)}")
                print(f"Traceback: {error_details}")
                return jsonify({'success': False, 'error': f'Error preparing time series data: {str(e)}'})

        elif model_type == 'classification':
            try:
                if feature_col and feature_col in df.columns:
                    X = clean_numeric_column(df[feature_col]).values.reshape(-1, 1)
                else:
                    X = np.arange(len(df)).reshape(-1, 1)

                y = df[target_col].copy()
                if y.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(y)

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                n_classes = len(np.unique(y))
                if n_classes == 2:
                    prediction_model = LogisticRegression(max_iter=1000)
                    model_name = "Binary Logistic Regression"
                else:
                    prediction_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
                    model_name = f"Random Forest ({n_classes} classes)"

                prediction_model.fit(X_scaled, y)
                y_pred = prediction_model.predict(X_scaled)
                accuracy = accuracy_score(y, y_pred)

                if hasattr(prediction_model, 'predict_proba'):
                    y_proba = prediction_model.predict_proba(X_scaled)
                    if n_classes == 2:
                        from sklearn.metrics import log_loss
                        logloss = log_loss(y, y_proba)
                        metrics = {
                            'accuracy': f"{accuracy * 100:.2f}%",
                            'log_loss': f"{logloss:.4f}",
                            'classes': str(n_classes),
                            'samples': len(y)
                        }
                    else:
                        metrics = {
                            'accuracy': f"{accuracy * 100:.2f}%",
                            'classes': str(n_classes),
                            'samples': len(y)
                        }
                else:
                    metrics = {
                        'accuracy': f"{accuracy * 100:.2f}%",
                        'classes': str(n_classes),
                        'samples': len(y)
                    }

                prediction_context.update({
                    'model_name': model_name,
                    'training_samples': len(y),
                    'num_classes': n_classes,
                    'accuracy': f"{accuracy * 100:.2f}%"
                })

            except Exception as e:
                return jsonify({'success': False, 'error': f'Error preparing classification data: {str(e)}'})

        elif model_type == 'clustering':
            try:
                if feature_col and feature_col in df.columns:
                    X = clean_numeric_column(df[feature_col]).values.reshape(-1, 1)
                else:
                    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                    if numeric_cols:
                        X = clean_numeric_column(df[numeric_cols[0]]).values.reshape(-1, 1)
                    else:
                        X = np.arange(len(df)).reshape(-1, 1)

                n_clusters = int(model_params.get('n_clusters', 3))
                prediction_model = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = prediction_model.fit_predict(X)

                from sklearn.metrics import silhouette_score
                if len(np.unique(clusters)) > 1:
                    silhouette_avg = silhouette_score(X, clusters)
                    metrics = {
                        'clusters': str(n_clusters),
                        'samples': len(X),
                        'silhouette_score': f"{silhouette_avg:.4f}",
                        'inertia': f"{prediction_model.inertia_:.2f}"
                    }
                else:
                    metrics = {
                        'clusters': str(n_clusters),
                        'samples': len(X),
                        'inertia': f"{prediction_model.inertia_:.2f}"
                    }

                model_name = f"K-Means Clustering ({n_clusters} clusters)"

                prediction_context.update({
                    'model_name': model_name,
                    'training_samples': len(X),
                    'n_clusters': n_clusters,
                    'cluster_centers': prediction_model.cluster_centers_.tolist()
                })

                df['cluster'] = clusters

            except Exception as e:
                return jsonify({'success': False, 'error': f'Error preparing clustering data: {str(e)}'})

        else:
            try:
                if feature_col and feature_col in df.columns:
                    X = clean_numeric_column(df[feature_col]).values.reshape(-1, 1)
                else:
                    X = np.arange(len(df)).reshape(-1, 1)

                y = df[target_col].values.astype(float)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                if len(df.columns) > 2 and feature_col:
                    other_features = [col for col in df.columns if col != target_col and col != feature_col]
                    if other_features:
                        X_sm = df[other_features].copy()
                        for col in other_features:
                            X_sm[col] = clean_numeric_column(X_sm[col])
                        X_sm = sm.add_constant(X_sm)
                        model_sm = sm.OLS(y, X_sm).fit()

                        prediction_model = {
                            'type': 'multiple_regression',
                            'sklearn_model': LinearRegression().fit(X_sm.drop('const', axis=1), y),
                            'statsmodels_model': model_sm,
                            'features': other_features
                        }
                        model_name = "Multiple Linear Regression"

                        summary_dict = {
                            'r_squared': f"{model_sm.rsquared:.4f}",
                            'adj_r_squared': f"{model_sm.rsquared_adj:.4f}",
                            'f_statistic': f"{model_sm.fvalue:.2f}",
                            'f_pvalue': f"{model_sm.f_pvalue:.4f}",
                            'aic': f"{model_sm.aic:.2f}",
                            'bic': f"{model_sm.bic:.2f}"
                        }

                        coefficients = []
                        for idx, param in enumerate(model_sm.params.index):
                            coefficients.append({
                                'feature': param,
                                'coef': f"{model_sm.params[idx]:.4f}",
                                'std_err': f"{model_sm.bse[idx]:.4f}",
                                't_value': f"{model_sm.tvalues[idx]:.4f}",
                                'p_value': f"{model_sm.pvalues[idx]:.4f}"
                            })

                        metrics = {
                            'r2_score': f"{model_sm.rsquared:.4f}",
                            'adj_r2': f"{model_sm.rsquared_adj:.4f}",
                            'f_statistic': f"{model_sm.fvalue:.2f}",
                            'samples': len(y),
                            'coefficients': coefficients
                        }
                    else:
                        prediction_model = LinearRegression()
                        prediction_model.fit(X_scaled, y)
                        model_name = "Linear Regression"
                else:
                    prediction_model = LinearRegression()
                    prediction_model.fit(X_scaled, y)
                    model_name = "Linear Regression"

                if model_name == "Linear Regression":
                    y_pred = prediction_model.predict(X_scaled)
                    mse = mean_squared_error(y, y_pred)
                    mae = mean_absolute_error(y, y_pred)
                    r2 = r2_score(y, y_pred)

                    metrics = {
                        'mse': f"{mse:.4f}",
                        'rmse': f"{np.sqrt(mse):.4f}",
                        'mae': f"{mae:.4f}",
                        'r2_score': f"{r2:.4f}",
                        'samples': len(y)
                    }

                prediction_context.update({
                    'model_name': model_name,
                    'training_samples': len(y),
                    'r2_score': metrics.get('r2_score', f"{r2:.4f}"),
                    'prediction_units': 'units'
                })

            except Exception as e:
                return jsonify({'success': False, 'error': f'Error preparing regression data: {str(e)}'})

        try:
            if model_type == 'timeseries':
                plot_data = create_visualization(df, feature_col, target_col,
                                                 y_pred if 'y_pred' in locals() else None,
                                                 model_type)
            elif model_type == 'clustering':
                plot_data = create_visualization(df, feature_col, target_col,
                                                 model_type=model_type,
                                                 clusters=df['cluster'] if 'cluster' in df.columns else None)
            else:
                plot_data = create_visualization(df, feature_col, target_col,
                                                 y_pred if 'y_pred' in locals() else None,
                                                 model_type)
        except Exception as e:
            print(f"Error creating visualization: {e}")
            plot_data = None

        model_trained = True

        serializable_prediction_context = {}
        for key, value in prediction_context.items():
            serializable_prediction_context[key] = convert_to_json_serializable(value)

        return jsonify({
            'success': True,
            'model_name': model_name,
            'model_type': model_type,
            'metrics': metrics,
            'plot': plot_data if plot_data else '',
            'prediction_context': serializable_prediction_context,
            'message': f'Model trained successfully with {len(df)} samples. Ready to predict {target_col}.'
        })

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in train_model: {str(e)}")
        print(f"Traceback: {error_details}")
        return jsonify({'success': False, 'error': f'Training error: {str(e)}'})


@app.route('/predict', methods=['POST'])
def predict():
    global prediction_model, model_trained, reference_data, model_type, column_info, scaler, prediction_context

    try:
        if not model_trained:
            return jsonify({'success': False, 'error': 'Model not trained yet. Please train the model first.'})

        data = request.get_json()
        input_value = data.get('feature_value', '').strip()

        if not input_value and model_type != 'clustering':
            return jsonify({'success': False, 'error': 'Please enter a value for prediction'})

        if model_type == 'timeseries':
            try:
                steps = int(input_value) if input_value.isdigit() else 1
                if steps < 1 or steps > 100:
                    return jsonify({'success': False, 'error': 'Please enter a number between 1 and 100 for steps'})

                if prediction_model['model_type'] in ['AR', 'ARIMA', 'ARMA', 'MA']:
                    forecast = prediction_model['model'].forecast(steps=steps)
                    predictions = [float(x) for x in forecast.tolist()]
                else:
                    trend = prediction_model.get('trend', 0)
                    last_value = prediction_model.get('last_value', 0)
                    predictions = [last_value + trend * (i + 1) for i in range(steps)]

                last_date = None
                if 'last_date' in prediction_model and prediction_model['last_date']:
                    try:
                        last_date = pd.to_datetime(prediction_model['last_date'])
                    except:
                        last_date = datetime.now()

                future_dates = []
                if last_date:
                    for i in range(steps):
                        if last_date.month == 12:
                            future_date = last_date.replace(year=last_date.year + 1, month=1, day=1)
                        else:
                            future_date = last_date.replace(month=last_date.month + 1, day=1)
                        future_dates.append(future_date.strftime('%m/%d/%Y'))
                        last_date = future_date
                else:
                    future_dates = [f"Period {i + 1}" for i in range(steps)]

                results = []
                for i, pred in enumerate(predictions):
                    results.append({
                        'period': i + 1,
                        'date': future_dates[i],
                        'prediction': float(pred),
                        'formatted': f"{pred:.2f}"
                    })

                stats = {
                    'average': f"{np.mean(predictions):.2f}",
                    'minimum': f"{np.min(predictions):.2f}",
                    'maximum': f"{np.max(predictions):.2f}",
                    'trend': 'increasing' if len(predictions) > 1 and predictions[-1] > predictions[0] else 'decreasing'
                }

                return jsonify({
                    'success': True,
                    'model_type': model_type,
                    'predictions': results,
                    'statistics': stats,
                    'steps': steps,
                    'message': f'Generated {steps} future predictions'
                })

            except Exception as e:
                return jsonify({'success': False, 'error': f'Time series prediction error: {str(e)}'})

        elif model_type == 'classification':
            try:
                feature_num = float(input_value)
                if scaler:
                    X_input = scaler.transform([[feature_num]])
                else:
                    X_input = [[feature_num]]

                prediction = int(prediction_model.predict(X_input)[0])

                if hasattr(prediction_model, 'predict_proba'):
                    probabilities = [float(x) for x in prediction_model.predict_proba(X_input)[0]]
                    result = {
                        'prediction': prediction,
                        'probabilities': probabilities,
                        'confidence': f"{max(probabilities) * 100:.1f}%"
                    }
                else:
                    result = {
                        'prediction': prediction
                    }

                return jsonify({
                    'success': True,
                    'model_type': model_type,
                    'input_value': input_value,
                    'result': result,
                    'message': f'Classification prediction: Class {prediction}'
                })

            except Exception as e:
                return jsonify({'success': False, 'error': f'Classification prediction error: {str(e)}'})

        elif model_type == 'clustering':
            try:
                if input_value:
                    feature_num = float(input_value)
                    X_input = [[feature_num]]
                    cluster = int(prediction_model.predict(X_input)[0])

                    centers = prediction_model.cluster_centers_
                    distances = [abs(feature_num - center[0]) for center in centers]
                    closest_center = centers[cluster][0]

                    result = {
                        'cluster': cluster,
                        'center_value': float(closest_center),
                        'distance_to_center': float(distances[cluster]),
                        'all_centers': [float(c[0]) for c in centers]
                    }

                    return jsonify({
                        'success': True,
                        'model_type': model_type,
                        'input_value': input_value,
                        'result': result,
                        'message': f'Data point assigned to Cluster {cluster}'
                    })
                else:
                    centers = prediction_model.cluster_centers_
                    result = {
                        'n_clusters': len(centers),
                        'centers': [float(c[0]) for c in centers],
                        'inertia': float(prediction_model.inertia_)
                    }

                    return jsonify({
                        'success': True,
                        'model_type': model_type,
                        'result': result,
                        'message': f'Clustering results: {len(centers)} clusters identified'
                    })

            except Exception as e:
                return jsonify({'success': False, 'error': f'Clustering prediction error: {str(e)}'})

        else:
            try:
                feature_num = float(input_value)
                if scaler:
                    X_input = scaler.transform([[feature_num]])
                else:
                    X_input = [[feature_num]]

                if isinstance(prediction_model, dict) and prediction_model.get('type') == 'multiple_regression':
                    prediction = float(prediction_model['sklearn_model'].predict(X_input)[0])
                else:
                    prediction = float(prediction_model.predict(X_input)[0])

                result = {
                    'prediction': prediction,
                    'prediction_formatted': f"{prediction:.2f}"
                }

                return jsonify({
                    'success': True,
                    'model_type': model_type,
                    'input_value': input_value,
                    'result': result,
                    'message': f'Regression prediction: {prediction:.2f}'
                })

            except Exception as e:
                return jsonify({'success': False, 'error': f'Regression prediction error: {str(e)}'})

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in predict: {str(e)}")
        print(f"Traceback: {error_details}")
        return jsonify({'success': False, 'error': f'Prediction error: {str(e)}'})


@app.route('/analyze_data', methods=['POST'])
def analyze_data():
    global reference_data

    try:
        if reference_data is None:
            return jsonify({'success': False, 'error': 'No data uploaded. Please upload data first.'})

        df = reference_data['df'].copy()

        analysis = {
            'data_shape': {
                'rows': len(df),
                'columns': len(df.columns)
            },
            'column_stats': {},
            'data_types': {},
            'correlation_matrix': None
        }

        for col in df.columns:
            analysis['data_types'][col] = str(df[col].dtype)

        numeric_cols = []
        for col in df.columns:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_cols.append(col)
                    clean_col = clean_numeric_column(df[col])
                    if clean_col.notna().sum() > 0:
                        analysis['column_stats'][col] = {
                            'mean': f"{clean_col.mean():.2f}",
                            'std': f"{clean_col.std():.2f}",
                            'min': f"{clean_col.min():.2f}",
                            'max': f"{clean_col.max():.2f}",
                            'median': f"{clean_col.median():.2f}",
                            'missing': int(clean_col.isna().sum()),
                            'non_missing': int(clean_col.notna().sum())
                        }
                else:
                    analysis['column_stats'][col] = {
                        'unique_values': int(df[col].nunique()),
                        'missing': int(df[col].isna().sum()),
                        'non_missing': int(df[col].notna().sum())
                    }
            except:
                analysis['column_stats'][col] = {
                    'error': 'Cannot calculate statistics'
                }

        if len(numeric_cols) > 1:
            numeric_df = df[numeric_cols].apply(clean_numeric_column)
            correlation = numeric_df.corr()
            analysis['correlation_matrix'] = correlation.to_dict()

        return jsonify({
            'success': True,
            'analysis': analysis,
            'message': f'Data analysis completed for {len(df)} rows and {len(df.columns)} columns'
        })

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in analyze_data: {str(e)}")
        print(f"Traceback: {error_details}")
        return jsonify({'success': False, 'error': f'Analysis error: {str(e)}'})


@app.route('/generate_report', methods=['POST'])
def generate_report():
    global reference_data, prediction_model, model_type, column_info, prediction_context

    try:
        if not model_trained:
            return jsonify({'success': False, 'error': 'Model not trained yet. Please train the model first.'})

        data = request.get_json()
        report_type = data.get('report_type', 'full')
        include_explanation = data.get('include_explanation', True)

        report = {
            'project_title': 'Predictive analysis using statistical and learning indicators',
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_info': {
                'model_type': model_type,
                'model_name': prediction_context.get('model_name', 'Unknown'),
                'target_variable': column_info.get('target', 'Unknown'),
                'feature_variable': column_info.get('feature', 'None'),
                'training_samples': prediction_context.get('training_samples', 0)
            },
            'data_summary': {
                'source': reference_data.get('filename', 'Unknown'),
                'rows': len(reference_data['df']),
                'columns': len(reference_data['df'].columns)
            },
            'results': {},
            'explanations': {},
            'recommendations': {}
        }

        if model_type == 'timeseries':
            report['results']['time_series'] = {
                'model_used': prediction_model.get('model_type', 'Unknown'),
                'trend': prediction_model.get('trend', 0),
                'last_value': prediction_model.get('last_value', 0),
                'forecast_horizon': 'Monthly'
            }

            if include_explanation:
                report['explanations']['time_series'] = """
                         A time series model was used to predict future values based on historical patterns.
                         This model is particularly useful for financial and economic data, as patterns tend to repeat over time.
                        These predictions can be used for financial planning and risk management in banking institutions.
                        """

        elif model_type == 'regression':
            report['results']['regression'] = {
                'model_used': 'Linear/Multiple Regression',
                'r_squared': prediction_context.get('r2_score', 'N/A')
            }

            report['explanations']['regression'] = """
                            A linear regression model was used to understand the relationship between independent and dependent variables.
                            This model helps identify the factors that most influence financial or banking results,
                            enabling organizations to make informed, data-driven decisions.
                            """

        elif model_type == 'classification':
            report['results']['classification'] = {
                'model_used': prediction_context.get('model_name', 'Unknown'),
                'accuracy': prediction_context.get('accuracy', 'N/A'),
                'classes': prediction_context.get('num_classes', 0)
            }

            if include_explanation:
                report['explanations']['classification'] = """
                            The classification model was used to divide data into specific categories based on their characteristics.
                            This model is useful for assessing credit risk, classifying customers, or identifying investment opportunities.
                            It can be used in banks to identify high-risk or low-risk customers.
                            """

        elif model_type == 'clustering':
            report['results']['clustering'] = {
                'model_used': 'K-Means Clustering',
                'n_clusters': prediction_context.get('n_clusters', 0),
                'cluster_centers': prediction_context.get('cluster_centers', [])
            }

            if include_explanation:
                report['explanations']['clustering'] = """
                            The clustering model was used to divide data into similar groups without prior knowledge of the categories.
                            This model is useful for segmenting customers based on their behavior or demographic characteristics,
                            which helps in designing targeted marketing campaigns and improving banking services.
                            """

        if model_type == 'timeseries':
            report['recommendations'] = [
                "Monitoring seasonal trends and patterns to predict the future",
                "Using forecasts for long-term financial planning",
                "Update the model regularly as new data becomes available."
            ]
        elif model_type == 'regression':
            report['recommendations'] = [
                "Focusing on the variables that most influence decision-making",
                "Using the model to assess the impact of new policies",
                "Conducting a sensitivity analysis to understand changes in results"
            ]
        elif model_type == 'classification':
            report['recommendations'] = [
                "Using the model to assess credit risk",
                "Developing different strategies for each customer segment",
                "Monitoring the accuracy of the model and updating it regularly"
            ]
        elif model_type == 'clustering':
            report['recommendations'] = [
                "Developing customized offers for each customer group",
                "Using the results to improve the customer experience.",
                "Re-evaluating groups regularly as customer behavior changes"
            ]

        return jsonify({
            'success': True,
            'report': report,
            'report_type': report_type,
            'message': 'The report was created successfully'
        })

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in generate_report: {str(e)}")
        print(f"Traceback: {error_details}")
        return jsonify({'success': False, 'error': f'Error generating report: {str(e)}'})


@app.route('/compare_models', methods=['POST'])
def compare_models():
    global reference_data, column_info

    try:
        if reference_data is None:
            return jsonify({'success': False, 'error': 'No data uploaded. Please upload data first.'})

        data = request.get_json()
        models_to_compare = data.get('models', ['regression', 'classification'])

        df = reference_data['df'].copy()
        feature_col = column_info['feature'] if column_info else None
        target_col = column_info['target'] if column_info else None

        if not target_col:
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            if numeric_cols:
                target_col = numeric_cols[0]
            else:
                return jsonify({'success': False, 'error': 'No suitable target column found for comparison'})

        df[target_col] = clean_numeric_column(df[target_col])
        df = df.dropna(subset=[target_col])

        if len(df) < 10:
            return jsonify({'success': False, 'error': 'Need at least 10 rows for model comparison'})

        comparison_results = {}

        if feature_col and feature_col in df.columns:
            X = clean_numeric_column(df[feature_col]).values.reshape(-1, 1)
        else:
            X = np.arange(len(df)).reshape(-1, 1)

        y = df[target_col].values

        if len(df) >= 20:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            X_train_scaled = StandardScaler().fit_transform(X_train)
            X_test_scaled = StandardScaler().fit_transform(X_test)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
            X_train_scaled = StandardScaler().fit_transform(X_train)
            X_test_scaled = X_train_scaled

        for model_name in models_to_compare:
            try:
                if model_name == 'regression':
                    model = LinearRegression()
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)

                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    comparison_results['linear_regression'] = {
                        'model_type': 'regression',
                        'mse': f"{mse:.4f}",
                        'rmse': f"{np.sqrt(mse):.4f}",
                        'r2_score': f"{r2:.4f}",
                        'complexity': 'Low',
                        'interpretability': 'High'
                    }

                elif model_name == 'classification':
                    unique_vals = len(np.unique(y))
                    if unique_vals <= 10:
                        if unique_vals == 2:
                            model = LogisticRegression(max_iter=1000)
                            model_name_display = 'logistic_regression'
                        else:
                            model = RandomForestClassifier(n_estimators=50, random_state=42)
                            model_name_display = 'random_forest_classifier'

                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        accuracy = accuracy_score(y_test, y_pred)

                        comparison_results[model_name_display] = {
                            'model_type': 'classification',
                            'accuracy': f"{accuracy * 100:.2f}%",
                            'n_classes': unique_vals,
                            'complexity': 'Medium',
                            'interpretability': 'Medium'
                        }

                elif model_name == 'timeseries':
                    try:
                        model_ar = AutoReg(y, lags=1).fit()
                        aic_ar = model_ar.aic

                        comparison_results['ar_model'] = {
                            'model_type': 'timeseries',
                            'aic': f"{aic_ar:.2f}",
                            'lags': 1,
                            'complexity': 'Medium',
                            'interpretability': 'High'
                        }
                    except:
                        pass

                elif model_name == 'clustering':
                    inertias = []
                    max_k = min(10, len(df) // 2)

                    if max_k > 1:
                        for k in range(2, max_k + 1):
                            kmeans = KMeans(n_clusters=k, random_state=42)
                            kmeans.fit(X)
                            inertias.append(kmeans.inertia_)

                        if len(inertias) > 1:
                            optimal_k = 3

                            comparison_results['kmeans_clustering'] = {
                                'model_type': 'clustering',
                                'optimal_k': optimal_k,
                                'inertia': f"{inertias[optimal_k - 2]:.2f}" if optimal_k - 2 < len(inertias) else 'N/A',
                                'complexity': 'Low',
                                'interpretability': 'Medium'
                            }

            except Exception as model_error:
                print(f"Error testing {model_name}: {model_error}")
                continue

        if comparison_results:
            best_model = None
            best_score = -float('inf')

            for model_name, results in comparison_results.items():
                if results['model_type'] == 'regression' and 'r2_score' in results:
                    score = float(results['r2_score'])
                elif results['model_type'] == 'classification' and 'accuracy' in results:
                    score = float(results['accuracy'].replace('%', ''))
                elif results['model_type'] == 'timeseries' and 'aic' in results:
                    score = -float(results['aic'])
                else:
                    score = -float('inf')

                if score > best_score:
                    best_score = score
                    best_model = model_name

            comparison_summary = {
                'total_models_tested': len(comparison_results),
                'best_model': best_model,
                'best_score': f"{best_score:.4f}",
                'recommendation': f"The {best_model} model is the most suitable for the current data",
                'models_comparison': comparison_results
            }

            return jsonify({
                'success': True,
                'comparison': comparison_summary,
                'message': f'The {len(comparison_results)} model was successfully compared'
            })
        else:
            return jsonify({'success': False, 'error': 'The system was unable to test any model'})

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in compare_models: {str(e)}")
        print(f"Traceback: {error_details}")
        return jsonify({'success': False, 'error': f'Comparison error: {str(e)}'})


@app.route('/export_notebook', methods=['POST'])
def export_notebook():
    global reference_data, prediction_model, model_type, column_info, prediction_context

    try:
        if not reference_data or not model_trained:
            return jsonify({'success': False, 'error': 'No trained model available for export'})

        data = request.get_json()
        export_format = data.get('format', 'python')

        df = reference_data['df']
        feature_col = column_info['feature'] if column_info else None
        target_col = column_info['target'] if column_info else None

        python_code = f'''# -*- coding: utf-8 -*-
"""
Python Notebook file for predictive analysis
Created by the Integrated Banking Analysis System
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

# =========================================================================
# 1. Importing the necessary libraries
# =========================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score


'''

        if model_type == 'timeseries':
            python_code += '''
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
'''
        elif model_type == 'classification':
            python_code += '''
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
'''
        elif model_type == 'clustering':
            python_code += '''
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
'''
        else:
            python_code += '''
from sklearn.linear_model import LinearRegression, Ridge
import statsmodels.api as sm
'''

        python_code += f'''
# =========================================================================
# 2. Data Loading
# =========================================================================
data = {df.head(100).to_dict('list')}

df = pd.DataFrame(data)
print("Data shape:", df.shape)
print("\\nColumns:", df.columns.tolist())

# =========================================================================
# 3. Exploratory Data Analysis
# =========================================================================

print("\\n=== Exploratory Analysis ===")

print("\\nDescriptive statistics:")
print(df.describe())

print("\\nMissing values:")
print(df.isnull().sum())

# =========================================================================
# 4. Data Preparation
# =========================================================================

def clean_numeric_column(series):
    if series.dtype == 'object':
        cleaned = series.astype(str).str.replace(',', '').str.replace('$', '').str.replace('%', '')
        return pd.to_numeric(cleaned, errors='coerce')
    return pd.to_numeric(series, errors='coerce')

for col in df.columns:
    if df[col].dtype == 'object':
        try:
            df[col] = clean_numeric_column(df[col])
        except:
            pass

feature_col = '{feature_col if feature_col else "None"}'
target_col = '{target_col if target_col else "None"}'

print(f"\\n Independent variable (Feature): {{feature_col}}")
print(f"Dependent variable (Target): {{target_col}}")

# =========================================================================
# 5. Building and Training the Model
# =========================================================================
'''

        if model_type == 'timeseries':
            python_code += f'''
if '{feature_col}' in df.columns and pd.api.types.is_datetime64_any_dtype(df['{feature_col}']):
    df = df.sort_values('{feature_col}')
    dates = df['{feature_col}']
    series = df['{target_col}'].values

    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(series, order=(1, 1, 1))
    model_fit = model.fit()

   print("\\nARIMA Model Summary:")

print(model_fit.summary())
forecast = model_fit.forecast(steps=5)
print("\\nForecasts for the next 5 periods:")
print(forecast)

'''
        elif model_type == 'classification':
            python_code += f'''
X = df[['{feature_col if feature_col else "index"}']].values if '{feature_col}' in df.columns else np.arange(len(df)).reshape(-1, 1)
y = df['{target_col}'].values

from sklearn.preprocessing import LabelEncoder
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

n_classes = len(np.unique(y))
if n_classes == 2:
    model = LogisticRegression(max_iter=1000)
else:
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\\nModel accuracy: {{accuracy * 100:.2f}}%")

'''
        elif model_type == 'clustering':
            python_code += f'''
X = df[['{feature_col if feature_col else "index"}']].values if '{feature_col}' in df.columns else np.arange(len(df)).reshape(-1, 1)

n_clusters = 3 
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X)

# Adding clusters to the data
df['cluster'] = clusters

print(f"\\nCluster Centers:")
print(kmeans.cluster_centers_)
print(f"\\nDistributing the data across clusters:")
print(df['cluster'].value_counts())
'''
        else:
            python_code += f'''
X = df[['{feature_col if feature_col else "index"}']].values if '{feature_col}' in df.columns else np.arange(len(df)).reshape(-1, 1)
y = df['{target_col}'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\\nMean Squared Error: {{mse:.4f}}")
print(f"R-squared: {{r2:.4f}}")
print(f"\\nRegression coefficient: {{model.coef_[0]:.4f}}")
print(f"Constant: {{model.intercept_:.4f}}")
'''

        python_code += '''
# =========================================================================
# 6. Graphical Visualization
# =========================================================================

plt.figure(figsize=(10, 6))

'''

        if model_type == 'timeseries':
            python_code += f'''
if '{feature_col}' in df.columns and pd.api.types.is_datetime64_any_dtype(df['{feature_col}']):
    plt.plot(df['{feature_col}'], df['{target_col}'], 'b-', label='Historical data')
    plt.xlabel('the date')
else:
    plt.plot(df['{target_col}'].values, 'b-', label='Data')
plt.xlabel('time period')

plt.ylabel('{target_col}')
plt.title('Time analysis of {target_col}')
plt.legend()
plt.grid(True, alpha=0.3)
'''
        elif model_type == 'classification':
            python_code += f'''
plt.scatter(X_train_scaled, y_train, alpha=0.6, label='Training data')
plt.scatter(X_test_scaled, y_test, alpha=0.6, label='Test data', marker='x')
plt.xlabel('{feature_col if feature_col else "Independent variable"}')
plt.ylabel('{target_col}')
plt.title('Classification model')
plt.legend()
plt.grid(True, alpha=0.3)
'''
        elif model_type == 'clustering':
            python_code += f'''

plt.scatter(X, np.zeros_like(X), c=clusters, cmap='viridis', s=100)
plt.xlabel('{feature_col if feature_col else "Value"}')
plt.title('Clustering results (K-Means)')
plt.colorbar(label='المجموعة')
plt.grid(True, alpha=0.3)
'''
        else:
            python_code += f'''

plt.scatter(X_train_scaled, y_train, alpha=0.6, label='Training Data')
plt.scatter(X_test_scaled, y_test, alpha=0.6, label='Test data', marker='x')

x_line = np.linspace(X_train_scaled.min(), X_train_scaled.max(), 100).reshape(-1, 1)
y_line = model.predict(x_line)
plt.plot(x_line, y_line, 'r-', linewidth=2, label='Regression line')

plt.xlabel('{feature_col if feature_col else "Independent variable"}')
plt.ylabel('{target_col}')
plt.title('Linear regression model')
plt.legend()
plt.grid(True, alpha=0.3)
'''

        python_code += '''
plt.tight_layout()
plt.show()

# ============================================================================
# 7. Interpretation and Conclusions
# =========================================================================

print("\\n=== Conclusions and Recommendations ===")

'''

        if model_type == 'timeseries':
            python_code += '''
print("1. The time model can be used to predict future values")
print("2. It is recommended to update the model regularly as new data becomes available")
print("3. The forecasts can be used for financial planning and risk management")
'''
        elif model_type == 'classification':
            python_code += '''
print("1. The model is suitable for assessing credit risk and classifying customers")
print("2. It can be used to develop targeted marketing strategies")
print("3. It is recommended to monitor the accuracy of the model and update it regularly")
'''
        elif model_type == 'clustering':
            python_code += '''
print("1. The model helps in segmenting customers into homogeneous groups")
print("2. The results can be used to improve banking services offered")
print("3. It is advisable to update the segments as customer behavior changes")
'''
        else:
            python_code += '''
print("1. The model helps in understanding the relationships between variables")
print("2. It can be used to evaluate the impact of new policies")
print("3. Sensitivity analysis is recommended to understand changes in outcomes")
'''

        python_code += '''
# ============================================================================
# End of file
# =========================================================================
'''

        notebook_path = os.path.join(app.config['UPLOAD_FOLDER'], 'analysis_notebook.py')
        with open(notebook_path, 'w', encoding='utf-8') as f:
            f.write(python_code)

        return jsonify({
            'success': True,
            'download_url': '/download_notebook',
            'code_preview': python_code[:1000] + '...' if len(python_code) > 1000 else python_code,
            'message': 'The Python Notebook file was successfully created'
        })

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in export_notebook: {str(e)}")
        print(f"Traceback: {error_details}")
        return jsonify({'success': False, 'error': f'Export error: {str(e)}'})


@app.route('/download_notebook', methods=['GET'])
def download_notebook():
    try:
        notebook_path = os.path.join(app.config['UPLOAD_FOLDER'], 'analysis_notebook.py')
        if os.path.exists(notebook_path):
            return send_from_directory(app.config['UPLOAD_FOLDER'], 'analysis_notebook.py',
                                       as_attachment=True,
                                       download_name=f'financial_analysis_{datetime.now().strftime("%Y%m%d")}.py')
        else:
            return jsonify({'success': False, 'error': 'No notebook available'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Download error: {str(e)}'})


@app.route('/clear_data', methods=['POST'])
def clear_data():
    global reference_data, prediction_model, model_type, model_trained, column_info, time_series_info, scaler, prediction_context

    reference_data = None
    prediction_model = None
    model_type = None
    model_trained = False
    column_info = None
    time_series_info = None
    scaler = None
    prediction_context = {}

    return jsonify({
        'success': True,
        'message': 'All data and models cleared successfully'
    })


@app.route('/get_sample_data', methods=['GET'])
def get_sample_data():
    try:
        dates = pd.date_range(start='2023-01-01', periods=12, freq='MS').strftime('%m/%d/%Y').tolist()

        sample_data = {
            'Date': dates,
            'Customers_Thousands': [52.0, 53.5, 54.2, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0],
            'Loans_Millions': [900, 910, 920, 930, 940, 950, 960, 970, 980, 990, 1000, 1010],
            'Deposits_Millions': [1500, 1510, 1520, 1530, 1540, 1550, 1560, 1570, 1580, 1590, 1600, 1610],
            'Revenue_Thousands': [610, 625, 640, 656, 670, 685, 700, 716, 732, 748, 764, 780],
            'Profit_Margin_Percent': [15.2, 15.5, 15.8, 16.1, 16.4, 16.7, 17.0, 17.3, 17.6, 17.9, 18.2, 18.5]
        }

        df = pd.DataFrame(sample_data)

        sample_path = os.path.join(app.config['UPLOAD_FOLDER'], 'sample_banking_data.csv')
        df.to_csv(sample_path, index=False)

        preview = df.head().to_html(
            classes='table table-striped table-bordered',
            index=False,
            na_rep='NaN'
        )

        return jsonify({
            'success': True,
            'sample_data': sample_data,
            'preview': preview,
            'download_url': '/download_sample',
            'message': 'Sample bank statements were created'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': f'Error creating sample data: {str(e)}'})


@app.route('/download_sample', methods=['GET'])
def download_sample():
    try:
        sample_path = os.path.join(app.config['UPLOAD_FOLDER'], 'sample_banking_data.csv')
        if os.path.exists(sample_path):
            return send_from_directory(app.config['UPLOAD_FOLDER'], 'sample_banking_data.csv',
                                       as_attachment=True,
                                       download_name='sample_banking_data.csv')
        else:
            return jsonify({'success': False, 'error': 'Sample data not available'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Download error: {str(e)}'})


@app.route('/template')
def template():
    return render_template('index.html')


if __name__ == '__main__':
    print("=" * 70)
    print("Integrated Banking Predictive Analysis System")
    print("=" * 70)
    print("Features:")
    print("1. Supports all project models: AR, MA, ARMA, ARIMA")
    print("2. Machine learning models: Regression, Classification, Clustering")
    print("3. Generate analytical reports")
    print("4. Export Python Notebooks")
    print("5. Compare different models")
    print("6. Full Arabic interface")
    print("\n Access: http://127.0.0.1:5000/")
    print("=" * 70)
    app.run(debug=True, host='127.0.0.1', port=5000)
