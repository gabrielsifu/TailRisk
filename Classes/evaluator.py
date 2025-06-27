import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_score, recall_score, f1_score,
    matthews_corrcoef
)
import seaborn as sns


class Evaluator:
    def __init__(self, sett):
        self.sett = sett
        self.predictions = joblib.load('./data/predictions/predictions.joblib')
        self.n_seeds = 10

    def ensure_evaluation_directory(self):
        """Ensure the evaluation directory exists."""
        os.makedirs('data/evaluation/', exist_ok=True)

    def concatenate_predictions(self):
        """Concatenate all prediction DataFrames into one."""
        all_predictions = pd.concat(self.predictions.values(),
                                    keys=self.predictions.keys()).reset_index(level=[0], drop=True)
        all_predictions.reset_index(inplace=True)
        return all_predictions

    def compute_cross_entropy_loss(self, all_predictions):
        """Compute cross-entropy loss for each sample for each model."""
        epsilon = 1e-15  # To prevent log(0)
        model_columns = [col for col in all_predictions.columns if col.startswith('y_pred')]

        for model_col in model_columns:
            loss_col = f'loss_{model_col[:-3]}'
            loss = -(
                    all_predictions['y_true'] * np.log(np.clip(all_predictions[model_col], epsilon, 1 - epsilon)) +
                    (1 - all_predictions['y_true']) * np.log(
                np.clip(1 - all_predictions[model_col], epsilon, 1 - epsilon))
            )
            if loss_col not in all_predictions:
                all_predictions[loss_col] = 0
            all_predictions[loss_col] += loss / self.n_seeds
        return all_predictions

    def prepare_data(self, all_predictions):
        """Prepare the data by ensuring datetime format and sorting."""
        all_predictions['Date'] = pd.to_datetime(all_predictions['Date'])
        all_predictions.sort_values('Date', inplace=True)
        return all_predictions

    def compute_moving_average_loss(self, all_predictions):
        """Compute moving average of loss over a 30-day window for each model."""
        model_loss_columns = [col for col in all_predictions.columns if col.startswith('loss_y_pred')]
        all_predictions_grouped = all_predictions.groupby('Date').mean()
        loss_moving_avg = {}

        for loss_col in model_loss_columns:
            loss_moving_avg[loss_col] = all_predictions_grouped[loss_col].rolling(window=252, min_periods=252).mean()
        return loss_moving_avg

    def plot_moving_average_loss(self, loss_moving_avg):
        """Plot and save the one-year rolling average of cross-entropy loss over time for each model."""

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(10, 6))

        palette = sns.color_palette("colorblind")
        color_0 = palette[0]
        color_1 = palette[9]
        color_2 = palette[3]
        color_3 = palette[1]
        color_4 = palette[4]
        color_5 = palette[6]
        color_6 = (0, 0, 0)
        color_7 = palette[7]
        colors = [color_0, color_1, color_2, color_3, color_4, color_5, color_6, color_7]

        i = 0
        for (loss_col, series), color in zip(loss_moving_avg.items(), colors):
            i += 1
            if i % 2 == 0:
                line_style = '--'
            else:
                line_style = '-'
            label = loss_col.replace('loss_y_pred_', '').replace('_', ' ').title()
            plt.plot(series.index, series.values, label=label,
                     color=color, linewidth=2, alpha=0.7, linestyle=line_style)
        plt.title('One-Year Rolling Average of Cross-Entropy Loss', fontsize=18)
        plt.xlabel('Year', fontsize=16)
        plt.ylabel('Cross-Entropy Loss', fontsize=16)

        # Ativa o grid (mantido conforme seu pedido)
        plt.grid(True)

        # Legenda com caixinha branca sobre o grid
        legend = plt.legend(
            fontsize=14,
            loc='upper left',
            frameon=True,
            facecolor='white',
            edgecolor='gray',
            framealpha=1  # fundo completamente opaco
        )
        legend.get_frame().set_linewidth(0.5)

        # plt.tight_layout()
        plt.tick_params(axis='both', labelsize=14)
        plt.savefig('data/evaluation/loss_moving_average.png', dpi=300)
        # plt.savefig('data/evaluation/loss_moving_average.pdf', bbox_inches='tight')
        plt.close()

    def prepare_histogram_data(self, all_predictions):
        """Agrupa modelos com o mesmo nome base (sem o sufixo _01, _02, etc.),
        faz a média das predições e separa com base no y_true para plotagem de histogramas."""

        import re
        from collections import defaultdict

        # Pega as colunas que têm predições
        model_columns = [col for col in all_predictions.columns if col.startswith('y_pred')]

        # Agrupa por nome base (removendo sufixos como _01, _02, etc.)
        grouped_preds = defaultdict(list)
        for col in model_columns:
            base_name = re.sub(r'_\d+$', '', col)  # remove o "_01", "_02" do final
            grouped_preds[base_name].append(col)

        # Calcula a média das predições para cada grupo
        averaged_predictions = {}
        for base_name, cols in grouped_preds.items():
            averaged_predictions[base_name] = all_predictions[cols].mean(axis=1)

        # Separa as predições médias por classe verdadeira
        histograms = {}
        for base_name, preds in averaged_predictions.items():
            y_pred_0 = preds[all_predictions['y_true'] == 0]
            y_pred_1 = preds[all_predictions['y_true'] == 1]
            y_pred_0[y_pred_0>0.1] = 0.1
            y_pred_1[y_pred_1>0.1] = 0.1
            histograms[base_name] = (y_pred_0, y_pred_1)

        return histograms

    def plot_histograms(self, histograms):
        """Plot and save histograms of predicted values for each model."""

        os.makedirs('data/evaluation/histogram_predictions', exist_ok=True)

        plt.style.use('seaborn-v0_8-whitegrid')
        palette = sns.color_palette("colorblind")
        color_y0 = 'black'
        color_y1 = palette[3]  # Light red

        for model_col, (y_pred_0, y_pred_1) in histograms.items():
            fig, ax1 = plt.subplots(figsize=(10, 6))

            ax1.hist(y_pred_0, bins=50, alpha=1, label='y_true = 0', color=color_y0)
            ax1.set_xlabel('Predicted Value (y_pred)', fontsize=16)
            ax1.set_ylabel('Frequency (y_true = 0)', color=color_y0, fontsize=16)
            ax1.tick_params(axis='y', labelcolor=color_y0)
            ax1.tick_params(axis='both', labelsize=14)

            ax2 = ax1.twinx()
            ax2.hist(y_pred_1, bins=50, alpha=0.5, label='y_true = 1', color=color_y1)
            ax2.set_ylabel('Frequency (y_true = 1)', color=color_y1, fontsize=16)
            ax2.tick_params(axis='y', labelcolor=color_y1)
            ax2.tick_params(axis='both', labelsize=14)

            # Title
            model_name = model_col.replace('y_pred_', '')
            plt.title(f'Predicted Values by True Label - {model_name}', fontsize=18)

            # Combined legend
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            legend = plt.legend(handles1 + handles2, labels1 + labels2,
                                loc='upper right',
                                fontsize=14,
                                frameon=True,
                                facecolor='white',
                                edgecolor='gray',
                                framealpha=1)
            legend.get_frame().set_linewidth(0.5)

            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'data/evaluation/histogram_predictions/{model_name}.png', dpi=300)
            # plt.savefig(f'data/evaluation/histogram_predictions/{model_name}.pdf', bbox_inches='tight')
            plt.close()

    def plot_histograms2(self, histograms):
        """Plot and save all histograms of predicted values in a single figure with fixed axes."""
        import math
        import numpy as np

        os.makedirs('data/evaluation/histogram_predictions', exist_ok=True)

        plt.style.use('seaborn-v0_8-whitegrid')
        palette = sns.color_palette("colorblind")
        color_y0 = 'black'
        color_y1 = palette[3]  # Light red

        num_models = len(histograms)
        ncols = 2
        nrows = math.ceil(num_models / ncols)

        # Flatten all data to determine global x and y limits
        all_y0 = np.concatenate([y0 for y0, _ in histograms.values()])
        all_y1 = np.concatenate([y1 for _, y1 in histograms.values()])

        # Determine global x limits
        global_xmin = min(all_y0.min(), all_y1.min())
        global_xmax = max(all_y0.max(), all_y1.max())

        # Create shared bins across all plots
        bins = np.linspace(global_xmin, global_xmax, 51)

        # Determine max heights for y-axes
        y0_max = max(np.histogram(y0, bins=bins)[0].max() for y0, _ in histograms.values())
        y1_max = max(np.histogram(y1, bins=bins)[0].max() for _, y1 in histograms.values())

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 6 * nrows))
        axes = axes.flatten()

        for idx, (model_col, (y_pred_0, y_pred_1)) in enumerate(histograms.items()):
            ax1 = axes[idx]
            ax2 = ax1.twinx()
            ax2.grid(False)

            ax1.hist(y_pred_0, bins=bins, alpha=1, label='y_true = 0', color=color_y0)
            ax2.hist(y_pred_1, bins=bins, alpha=0.5, label='y_true = 1', color=color_y1)

            model_name = model_col.replace('y_pred_', '')
            ax1.set_title(f'{model_name}', fontsize=16)

            ax1.set_xlabel('Predicted Value', fontsize=12)
            ax1.set_ylabel('Freq (y=0)', color=color_y0, fontsize=12)
            ax2.set_ylabel('Freq (y=1)', color=color_y1, fontsize=12)

            ax1.tick_params(axis='y', labelcolor=color_y0)
            ax2.tick_params(axis='y', labelcolor=color_y1)

            # Set same limits for all plots
            ax1.set_xlim(global_xmin, global_xmax)
            ax1.set_ylim(0, y0_max)
            ax2.set_ylim(0, y1_max)

            if idx == 0:
                handles1, labels1 = ax1.get_legend_handles_labels()
                handles2, labels2 = ax2.get_legend_handles_labels()
                fig.legend(handles1 + handles2, labels1 + labels2,
                           loc='upper center', ncol=2, fontsize=14, frameon=True)

        # Hide any unused subplots
        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.suptitle('Histograms of Predicted Values by Model', fontsize=20)
        fig.savefig('data/evaluation/histogram_predictions/all_models_comparison.png', dpi=300)
        plt.savefig('data/evaluation/histogram_predictions/all_models_comparison.pdf', bbox_inches='tight')
        plt.close()

    def compute_confusion_matrix(self, all_predictions):
        """Compute and plot the confusion matrix for each model."""
        os.makedirs('data/evaluation/confusion_matrix', exist_ok=True)
        model_columns = [col for col in all_predictions.columns if col.startswith('y_pred')]

        for model_col in model_columns:
            # Binarize predictions based on threshold 0.5
            y_true = all_predictions['y_true']
            y_pred_binary = (all_predictions[model_col] >= self.sett.DataEngineer.quantile_threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred_binary)

            # Save confusion matrix as a table
            # cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
            model_name = model_col.replace('y_pred_', '')
            # cm_df.to_csv(f'data/evaluation/confusion_matrix_{model_name}.csv')

            # Plot confusion matrix using matplotlib
            fig, ax = plt.subplots(figsize=(6, 4))
            cax = ax.matshow(cm, cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix - {model_name}', y=1.1)
            fig.colorbar(cax)

            # Set axis labels
            plt.xlabel('Predicted')
            plt.ylabel('Actual')

            # Loop over data dimensions and create text annotations.
            for (i, j), z in np.ndenumerate(cm):
                ax.text(j, i, f'{z}', ha='center', va='center', color='red')

            plt.tight_layout()
            plt.savefig(f'data/evaluation/confusion_matrix/{model_name}.png')
            plt.close()

    def compute_confusion_matrices_combined(self, all_predictions):
        """Agrupa modelos com o mesmo nome base (sem o sufixo _01, _02, etc.),
        calcula a média das predições e plota todas as matrizes de confusão em um único gráfico com estilo unificado."""

        import os
        import math
        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt
        import re
        from collections import defaultdict
        from sklearn.metrics import confusion_matrix

        os.makedirs('data/evaluation/confusion_matrix', exist_ok=True)

        plt.style.use('seaborn-v0_8-whitegrid')
        palette = sns.color_palette("colorblind")

        # Agrupar colunas por nome base (removendo sufixo de seed tipo _01, _02, etc.)
        model_columns = [col for col in all_predictions.columns if col.startswith('y_pred')]
        grouped_preds = defaultdict(list)
        for col in model_columns:
            base_name = re.sub(r'_\d+$', '', col)
            grouped_preds[base_name].append(col)

        averaged_predictions = {}
        for base_name, cols in grouped_preds.items():
            averaged_predictions[base_name] = all_predictions[cols].mean(axis=1)

        num_models = len(averaged_predictions)
        ncols = 2
        nrows = math.ceil(num_models / ncols)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 5 * nrows))
        axes = axes.flatten()

        for idx, (base_name, preds) in enumerate(averaged_predictions.items()):
            y_true = all_predictions['y_true']
            y_pred_binary = (preds >= self.sett.DataEngineer.quantile_threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred_binary)

            ax = axes[idx]
            ax.matshow(cm, cmap=plt.cm.Blues)

            # Título e eixos
            ax.set_title(f'{base_name}', fontsize=16, pad=12)
            ax.set_xlabel('Predicted', fontsize=12)
            ax.set_ylabel('Actual', fontsize=12)
            ax.xaxis.set_label_position('bottom')
            ax.xaxis.tick_bottom()
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['0', '1'], fontsize=11)
            ax.set_yticklabels(['0', '1'], fontsize=11)

            # Números dentro das células
            for (i, j), value in np.ndenumerate(cm):
                ax.text(j, i, f'{value}', ha='center', va='center', color='red', fontsize=12)

        # Esconder subplots não usados
        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.suptitle('Confusion Matrices by Model (Averaged over Seeds)', fontsize=20)
        fig.subplots_adjust(hspace=0.3)

        # Salvar gráfico final
        fig.savefig('data/evaluation/confusion_matrix/all_models_confusion_matrices.png', dpi=300)
        fig.savefig('data/evaluation/confusion_matrix/all_models_confusion_matrices.pdf', bbox_inches='tight')
        plt.close()

    def compute_roc_auc(self, all_predictions):
        """Agrupa modelos com o mesmo nome base, calcula a média das predições por grupo e plota a curva ROC com cores definidas manualmente."""

        import os
        import re
        from collections import defaultdict
        from sklearn.metrics import roc_curve, roc_auc_score
        import matplotlib.pyplot as plt
        import seaborn as sns

        os.makedirs('data/evaluation/roc_curve', exist_ok=True)

        plt.style.use('seaborn-v0_8-whitegrid')

        y_true = all_predictions['y_true']

        # Agrupar colunas por nome base (sem _01, _02...)
        model_columns = [col for col in all_predictions.columns if col.startswith('y_pred')]
        grouped_preds = defaultdict(list)
        for col in model_columns:
            base_name = re.sub(r'_\d+$', '', col)
            grouped_preds[base_name].append(col)

        base_model_names = list(grouped_preds.keys())

        # Definindo paleta de cores fixa
        palette = sns.color_palette("colorblind")
        color_0 = palette[0]
        color_1 = palette[9]
        color_2 = palette[3]
        color_3 = palette[1]
        color_4 = palette[4]
        color_5 = palette[6]
        color_6 = (0, 0, 0)
        color_7 = palette[7]
        colors = [color_0, color_1, color_2, color_3, color_4, color_5, color_6, color_7]

        # Garantir que há cores suficientes
        while len(colors) < len(base_model_names):
            colors.append((0, 0, 0))  # repete preto se faltar

        plt.figure(figsize=(10, 6))

        for i, (base_name, cols) in enumerate(grouped_preds.items()):
            y_pred_avg = all_predictions[cols].mean(axis=1)
            fpr, tpr, _ = roc_curve(y_true, y_pred_avg)
            roc_auc = roc_auc_score(y_true, y_pred_avg)

            linestyle = '--' if i % 2 == 0 else '-'
            color = colors[i]
            label = base_name.replace('_', ' ').title()

            plt.plot(fpr, tpr, label=f'{label}',  #  (AUC = {roc_auc:.2f})
                     color=color, linewidth=2, alpha=0.5, linestyle=linestyle)

        # Linha diagonal (baseline)
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

        # Estilo do gráfico
        plt.title('ROC Curve by Model', fontsize=18)
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.tick_params(axis='both', labelsize=14)
        plt.grid(True)

        # Legenda com fundo branco e moldura
        legend = plt.legend(
            fontsize=14,
            loc='lower right',
            frameon=True,
            facecolor='white',
            edgecolor='gray',
            framealpha=1
        )
        legend.get_frame().set_linewidth(0.5)

        plt.tight_layout()
        plt.savefig('data/evaluation/roc_curve/combined_roc_curve.png', dpi=300)
        plt.close()

    def compute_classification_metrics(self, all_predictions):
        """
        Agrupa predições por modelo base (média das seeds),
        calcula AUC ROC, F1 Score e Log Loss,
        e salva em formato acadêmico (CSV e LaTeX).
        """
        import os
        import re
        from collections import defaultdict
        import pandas as pd
        from sklearn.metrics import (
            log_loss,
            f1_score,
            roc_auc_score
        )

        os.makedirs('data/evaluation/', exist_ok=True)

        y_true = all_predictions['y_true']
        model_columns = [col for col in all_predictions.columns if col.startswith('y_pred')]

        # Agrupa colunas por modelo base (sem _01, _02, etc.)
        grouped_preds = defaultdict(list)
        for col in model_columns:
            base_name = re.sub(r'_\d+$', '', col)  # Remove sufixo de seed (_01, _02, etc.)
            grouped_preds[base_name].append(col)

        results = []

        for base_name, cols in grouped_preds.items():
            y_pred_avg = all_predictions[cols].mean(axis=1)
            y_pred_binary = (y_pred_avg >= self.sett.DataEngineer.quantile_threshold).astype(int)

            # Remove o prefixo 'y_pred_' do nome do modelo
            clean_name = re.sub(r'^y_pred_', '', base_name)

            row = {
                'Model': clean_name.replace('_', ' ').title(),
                'AUC ROC': roc_auc_score(y_true, y_pred_avg),
                'F1 Score': f1_score(y_true, y_pred_binary),
                'Log Loss': log_loss(y_true, y_pred_avg, labels=[0, 1]),
            }

            results.append(row)

        df_metrics = pd.DataFrame(results)
        df_metrics = df_metrics.round(4)
        df_metrics.sort_values(by='Model', inplace=True)  # Ordena em ordem alfabética

        # Salva CSV delimitado por ponto e vírgula
        df_metrics.to_csv('data/evaluation/classification_metrics.csv', index=False, sep=';')

        # Salva em LaTeX com estilo acadêmico
        with open('data/evaluation/classification_metrics_table.tex', 'w') as f:
            f.write(df_metrics.to_latex(
                index=False,
                caption='Classification Metrics by Model',
                label='tab:classification_metrics',
                column_format='lrrr',
                float_format="%.4f"
            ))

    def plot_y(self, df):
        """Plot and save the time series of log returns with periods where y = 1 highlighted."""

        # Set 'Date' as the index
        df.set_index('Date', inplace=True)
        df = df.loc[:, ['log_returns', 'y_true']]

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(10, 6))

        # Paleta colorblind do seaborn
        palette = sns.color_palette("colorblind")
        log_returns_color = palette[0]
        y_highlight_color = palette[3]

        # Plot log returns
        plt.plot(df.index, df['log_returns'], label='Log Returns',
                 color=log_returns_color, linewidth=2, alpha=1)

        # Highlight periods where y_true == 1
        plt.fill_between(df.index,
                         df['log_returns'].min(), df['log_returns'].max(),
                         where=(df['y_true'] == 1),
                         color=y_highlight_color, alpha=0.5, label='Left-tail Event')

        # Títulos e eixos
        plt.title('Time Series of Log Returns with y Highlighted', fontsize=18)
        plt.xlabel('Year', fontsize=16)
        plt.ylabel('Log Return', fontsize=16)

        # Grade
        plt.grid(True)

        # Legenda formatada
        legend = plt.legend(
            fontsize=14,
            loc='upper left',
            frameon=True,
            facecolor='white',
            edgecolor='gray',
            framealpha=1
        )
        legend.get_frame().set_linewidth(0.5)

        # Ajustes finais
        # plt.tight_layout()
        plt.tick_params(axis='both', labelsize=14)
        plt.savefig('data/evaluation/log_returns_and_y.png', dpi=300)
        # plt.savefig('data/evaluation/log_returns_and_y.pdf', bbox_inches='tight')
        plt.close()

    def evaluate(self):
        """Run the full evaluation process."""
        self.ensure_evaluation_directory()
        all_predictions = self.concatenate_predictions()

        self.plot_y(all_predictions.copy())

        all_predictions = self.compute_cross_entropy_loss(all_predictions)
        all_predictions = self.prepare_data(all_predictions)

        loss_moving_avg = self.compute_moving_average_loss(all_predictions)
        self.plot_moving_average_loss(loss_moving_avg)
        # histograms = self.prepare_histogram_data(all_predictions)
        # self.plot_histograms2(histograms)

        # # Compute and plot confusion matrix for each model
        # # self.compute_confusion_matrices_combined(all_predictions)
        # Plot return distribution
        # self.plot_return_distribution(all_predictions)

        # Compute ROC AUC and plot ROC curve for each model
        self.compute_roc_auc(all_predictions)
        # Compute classification metrics for each model
        self.compute_classification_metrics(all_predictions)

        print("Evaluation completed and results saved in 'data/evaluation/' directory.")

    def plot_return_distribution(self, df):
        """Agrupa modelos por nome base, calcula a média das predições e plota as distribuições de log_returns
        separadas por classe de confusão (TP, TN, FP, FN) para cada modelo."""

        import os
        import re
        from collections import defaultdict
        import matplotlib.pyplot as plt
        import seaborn as sns

        os.makedirs('data/evaluation/histograms_returns/', exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("colorblind")

        # Agrupar colunas de predições por nome base
        pred_cols = [col for col in df.columns if col.startswith('y_pred_')]
        grouped_preds = defaultdict(list)
        for col in pred_cols:
            base_name = re.sub(r'_\d+$', '', col)
            grouped_preds[base_name].append(col)

        for thresh in [0.02]:
            for base_name, cols in grouped_preds.items():
                # Calcular a média das predições entre as seeds
                df[f'y_pred_{base_name}'] = df[cols].mean(axis=1)

                # Criar coluna com os rótulos de confusão
                y_pred_col = f'y_pred_{base_name}'
                confusion_label_col = f'confusion_label_{base_name}'
                df[confusion_label_col] = df.apply(
                    lambda row: self.confusion_label(row['y_true'], row[y_pred_col], thresh=thresh), axis=1)

                # Labels de confusão e setup dos plots
                labels = ['TN', 'FP', 'FN', 'TP']
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                axes = axes.flatten()

                for idx, label in enumerate(labels):
                    subset = df[df[confusion_label_col] == label]
                    data = subset['log_returns']
                    ax = axes[idx]

                    ax.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                    ax.set_title(f'{base_name} - {label} ({len(data)})', fontsize=14)

                    # Estatísticas
                    mean = data.mean()
                    std = data.std()
                    median = data.median()
                    q75 = data.quantile(0.75)
                    p90 = data.quantile(0.90)
                    p95 = data.quantile(0.95)

                    ax.axvline(mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {100 * mean:.2f}%')
                    ax.axvline(median, color='green', linestyle='dashed', linewidth=1,
                               label=f'Median: {100 * median:.2f}%')
                    ax.axvline(q75, color='blue', linestyle='dashed', linewidth=1, label=f'75th: {100 * q75:.2f}%')
                    ax.axvline(p90, color='cyan', linestyle='dashed', linewidth=1, label=f'90th: {100 * p90:.2f}%')
                    ax.axvline(p95, color='magenta', linestyle='dashed', linewidth=1, label=f'95th: {100 * p95:.2f}%')

                    ax.legend(fontsize=9)
                    ax.set_xlabel('Log Returns', fontsize=11)
                    ax.set_ylabel('Frequency', fontsize=11)

                plt.tight_layout()
                plt.suptitle(f'Log Return Distribution by Confusion Class – {base_name}', fontsize=18, y=1.02)
                plt.savefig(f'data/evaluation/histograms_returns/{base_name}_{thresh}.png', dpi=300,
                            bbox_inches='tight')
                plt.close()

    @staticmethod
    def confusion_label(y_true, y_pred, thresh=0.9):
        if y_true == 1 and y_pred > thresh:
            return 'TP'
        elif y_true == 0 and y_pred <= thresh:
            return 'TN'
        elif y_true == 0 and y_pred > thresh:
            return 'FP'
        elif y_true == 1 and y_pred <= thresh:
            return 'FN'
        else:
            return 'Unknown'
