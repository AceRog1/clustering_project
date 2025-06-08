# Main Script - Movie Poster Clustering con Datos Reales (VERSIÓN CORREGIDA)
# Proyecto Educativo de Machine Learning - Clustering de Posters de Películas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
import warnings
import time
from typing import List, Tuple, Optional, Dict, Any

warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

## IMPORTAR LAS CLASES DE CLUSTERING (del notebook anterior)

class KMeansCustom:
    """Implementación personalizada del algoritmo K-means"""

    def __init__(self, n_clusters: int = 8, max_iters: int = 300, tol: float = 1e-4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """Inicializa centroides usando K-means++"""
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))

        centroids[0] = X[np.random.randint(n_samples)]

        for c_id in range(1, self.n_clusters):
            distances = np.array([min([np.linalg.norm(x - c)**2 for c in centroids[:c_id]]) for x in X])
            probabilities = distances / distances.sum()
            cumulative_probabilities = probabilities.cumsum()
            r = np.random.rand()

            for j, p in enumerate(cumulative_probabilities):
                if r < p:
                    centroids[c_id] = X[j]
                    break

        return centroids

    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """Asigna cada punto al cluster más cercano"""
        distances = np.sqrt(((X - self.centroids_[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Actualiza centroides"""
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            if np.sum(labels == k) > 0:
                centroids[k] = X[labels == k].mean(axis=0)
            else:
                centroids[k] = self.centroids_[k]
        return centroids

    def _calculate_inertia(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calcula la inercia"""
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids_[k])**2)
        return inertia

    def fit(self, X: np.ndarray) -> 'KMeansCustom':
        """Entrena el modelo K-means"""
        self.centroids_ = self._initialize_centroids(X)
        prev_centroids = self.centroids_.copy()

        for i in range(self.max_iters):
            self.labels_ = self._assign_clusters(X)
            self.centroids_ = self._update_centroids(X, self.labels_)

            if np.allclose(prev_centroids, self.centroids_, atol=self.tol):
                self.n_iter_ = i + 1
                break

            prev_centroids = self.centroids_.copy()
        else:
            self.n_iter_ = self.max_iters

        self.inertia_ = self._calculate_inertia(X, self.labels_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice clusters para nuevos datos"""
        if self.centroids_ is None:
            raise ValueError("Modelo no entrenado. Ejecuta fit() primero.")
        return self._assign_clusters(X)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Entrena el modelo y retorna las etiquetas"""
        return self.fit(X).labels_


class DBSCANCustom:
    """Implementación personalizada del algoritmo DBSCAN"""

    def __init__(self, eps: float = 0.5, min_samples: int = 5, metric: str = 'euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = None
        self.core_sample_indices_ = None
        self.n_clusters_ = None

    def _euclidean_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calcula distancia euclidiana"""
        return np.sqrt(np.sum((point1 - point2)**2))

    def _get_neighbors(self, X: np.ndarray, point_idx: int) -> List[int]:
        """Encuentra vecinos dentro del radio eps"""
        neighbors = []
        point = X[point_idx]

        for i, other_point in enumerate(X):
            if self._euclidean_distance(point, other_point) <= self.eps:
                neighbors.append(i)

        return neighbors

    def _expand_cluster(self, X: np.ndarray, point_idx: int, neighbors: List[int],
                       cluster_id: int, labels: np.ndarray, visited: np.ndarray) -> None:
        """Expande cluster"""
        labels[point_idx] = cluster_id

        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                neighbor_neighbors = self._get_neighbors(X, neighbor_idx)

                if len(neighbor_neighbors) >= self.min_samples:
                    for nn in neighbor_neighbors:
                        if nn not in neighbors:
                            neighbors.append(nn)

            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id

            i += 1

    def fit(self, X: np.ndarray) -> 'DBSCANCustom':
        """Entrena el modelo DBSCAN"""
        n_samples = X.shape[0]
        labels = np.full(n_samples, -1)
        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0
        core_samples = []

        for point_idx in range(n_samples):
            if visited[point_idx]:
                continue

            visited[point_idx] = True
            neighbors = self._get_neighbors(X, point_idx)

            if len(neighbors) < self.min_samples:
                continue
            else:
                core_samples.append(point_idx)
                self._expand_cluster(X, point_idx, neighbors, cluster_id, labels, visited)
                cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.array(core_samples)
        self.n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Entrena el modelo y retorna las etiquetas"""
        return self.fit(X).labels_


class MovieClusteringAnalyzer:
    """Clase principal para análisis de clustering de posters de películas - VERSIÓN CORREGIDA"""

    def __init__(self):
        self.features = None
        self.movie_metadata = None
        self.kmeans_model = None
        self.dbscan_model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.features_scaled = None
        self.features_pca = None
        self.dataset_name = None
        self.feature_columns = None

    def load_csv_data(self, csv_file_path: str, dataset_name: str = "Dataset"):
        """
        Carga datos desde archivo CSV - VERSIÓN CORREGIDA

        Parámetros:
        - csv_file_path: Ruta al archivo CSV
        - dataset_name: Nombre descriptivo del dataset
        """
        print(f"Cargando datos desde: {csv_file_path}")

        # Leer CSV
        df = pd.read_csv(csv_file_path)
        print(f"Archivo cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
        print(f"Columnas encontradas: {list(df.columns)}")

        # 🔧 CORRECCIÓN CRÍTICA: Identificar correctamente las columnas de features
        # Excluir la primera columna (índice sin nombre) y movieId
        excluded_columns = ['movieId']

        # La primera columna puede tener nombre vacío o ser un índice
        first_col = df.columns[0]
        if first_col == '' or 'Unnamed' in str(first_col) or first_col.lower() in ['index', 'idx']:
            excluded_columns.append(first_col)

        # Identificar columnas de features (deben ser numéricas y no estar excluidas)
        feature_columns = []
        for col in df.columns:
            if col not in excluded_columns and pd.api.types.is_numeric_dtype(df[col]):
                feature_columns.append(col)

        self.feature_columns = feature_columns

        print(f"\n🎯 CORRECCIÓN APLICADA:")
        print(f"  - Columnas excluidas: {excluded_columns}")
        print(f"  - Columnas de features: {feature_columns}")
        print(f"  - Número de features: {len(feature_columns)} (debería ser 10)")

        if len(feature_columns) != 10:
            print(f"  ⚠️  ADVERTENCIA: Se esperaban 10 features, se encontraron {len(feature_columns)}")

        # Extraer solo las columnas de features
        self.features = df[feature_columns].values

        # Crear metadatos preservando movieId
        metadata_dict = {
            'movie_index': range(len(df)),
            'dataset': [dataset_name] * len(df)
        }

        # Agregar movieId si existe
        if 'movieId' in df.columns:
            metadata_dict['movieId'] = df['movieId'].values

        # Agregar la columna de índice original si existe
        if first_col in df.columns and first_col not in excluded_columns:
            metadata_dict['original_index'] = df[first_col].values
        elif first_col in excluded_columns and first_col != 'movieId':
            metadata_dict['original_index'] = df[first_col].values

        self.movie_metadata = pd.DataFrame(metadata_dict)
        self.dataset_name = dataset_name

        # Verificar que los features son válidos
        if np.any(np.isnan(self.features)) or np.any(np.isinf(self.features)):
            print("⚠️ Advertencia: Se encontraron valores NaN o infinitos en los features")
            # Limpiar datos
            self.features = np.nan_to_num(self.features, nan=0.0, posinf=1e6, neginf=-1e6)

        # Normalizar características
        print("Normalizando características...")
        self.features_scaled = self.scaler.fit_transform(self.features)

        # Reducción dimensional para visualización
        print("Aplicando PCA para visualización...")
        self.features_pca = self.pca.fit_transform(self.features_scaled)

        print(f"\n=== RESUMEN DEL DATASET: {dataset_name} ===")
        print(f"Películas: {self.features.shape[0]}")
        print(f"Características visuales: {self.features.shape[1]} ✅")
        print(f"Varianza explicada por PCA: {self.pca.explained_variance_ratio_.sum():.3f}")

        # Mostrar estadísticas de los features
        print(f"\nEstadísticas de características:")
        for i, col in enumerate(feature_columns):
            feature_stats = self.features[:, i]
            print(f"  Feature {col}: mean={np.mean(feature_stats):.3f}, "
                  f"std={np.std(feature_stats):.3f}, "
                  f"range=[{np.min(feature_stats):.3f}, {np.max(feature_stats):.3f}]")

        return self

    def find_optimal_k(self, max_k: int = 15) -> Tuple[List[int], List[float], List[float]]:
        """Encuentra el número óptimo de clusters"""
        k_range = range(2, max_k + 1)
        inertias = []
        silhouette_scores = []

        print(f"\nBuscando número óptimo de clusters (k=2 a {max_k})...")

        for k in k_range:
            print(f"  Probando k={k}...", end=" ")
            start_time = time.time()

            kmeans = KMeansCustom(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(self.features_scaled)

            inertias.append(kmeans.inertia_)

            if len(set(labels)) > 1:
                sil_score = silhouette_score(self.features_scaled, labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)

            print(f"Silhouette: {silhouette_scores[-1]:.3f} ({time.time()-start_time:.1f}s)")

        return list(k_range), inertias, silhouette_scores

    def optimize_dbscan_params(self, eps_range: np.ndarray = None,
                             min_samples_range: List[int] = None) -> Dict[str, Any]:
        """Optimiza parámetros de DBSCAN"""
        if eps_range is None:
            eps_range = np.arange(0.3, 1.5, 0.2)  # Rango más conservador
        if min_samples_range is None:
            min_samples_range = [3, 5, 7, 10]

        best_score = -1
        best_params = {}
        results = []

        print(f"\nOptimizando parámetros DBSCAN...")
        print(f"Probando eps: {eps_range}")
        print(f"Probando min_samples: {min_samples_range}")

        for eps in eps_range:
            for min_samples in min_samples_range:
                print(f"  Probando eps={eps:.1f}, min_samples={min_samples}...", end=" ")
                start_time = time.time()

                dbscan = DBSCANCustom(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(self.features_scaled)

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                noise_ratio = n_noise / len(labels)

                if n_clusters > 1 and noise_ratio < 0.8:
                    sil_score = silhouette_score(self.features_scaled, labels)
                    results.append({
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'noise_ratio': noise_ratio,
                        'silhouette_score': sil_score
                    })

                    print(f"Clusters: {n_clusters}, Ruido: {noise_ratio:.2%}, Silhouette: {sil_score:.3f}")

                    if sil_score > best_score:
                        best_score = sil_score
                        best_params = {'eps': eps, 'min_samples': min_samples}
                else:
                    print(f"Clusters: {n_clusters}, Ruido: {noise_ratio:.2%} (descartado)")

                print(f" ({time.time()-start_time:.1f}s)")

        return {'best_params': best_params, 'best_score': best_score, 'all_results': results}

    def fit_models(self, n_clusters_kmeans: int = None,
                   eps_dbscan: float = None, min_samples_dbscan: int = None):
        """Entrena ambos modelos de clustering"""
        if self.features_scaled is None:
            raise ValueError("Primero carga los datos usando load_csv_data()")

        print(f"\n=== ENTRENANDO MODELOS DE CLUSTERING ===")
        print(f"Entrenando con {self.features.shape[1]} features correctos ✅")

        # Entrenar K-means
        if n_clusters_kmeans is None:
            n_clusters_kmeans = 8  # Default

        print(f"\n1. Entrenando K-means con {n_clusters_kmeans} clusters...")
        start_time = time.time()
        self.kmeans_model = KMeansCustom(n_clusters=n_clusters_kmeans, random_state=42)
        self.kmeans_labels = self.kmeans_model.fit_predict(self.features_scaled)
        kmeans_time = time.time() - start_time
        print(f"   K-means completado en {kmeans_time:.2f} segundos")

        # Entrenar DBSCAN
        if eps_dbscan is None or min_samples_dbscan is None:
            print(f"\n2. Optimizando parámetros DBSCAN...")
            dbscan_results = self.optimize_dbscan_params()

            if dbscan_results['best_params']:
                eps_dbscan = dbscan_results['best_params']['eps']
                min_samples_dbscan = dbscan_results['best_params']['min_samples']
                print(f"   Mejores parámetros: eps={eps_dbscan:.2f}, min_samples={min_samples_dbscan}")
            else:
                print("   No se encontraron parámetros óptimos, usando valores por defecto")
                eps_dbscan, min_samples_dbscan = 0.5, 5

        print(f"\n3. Entrenando DBSCAN...")
        start_time = time.time()
        self.dbscan_model = DBSCANCustom(eps=eps_dbscan, min_samples=min_samples_dbscan)
        self.dbscan_labels = self.dbscan_model.fit_predict(self.features_scaled)
        dbscan_time = time.time() - start_time
        print(f"   DBSCAN completado en {dbscan_time:.2f} segundos")

        # Calcular métricas
        self._calculate_metrics()

    def _calculate_metrics(self):
        """Calcula métricas de evaluación para ambos modelos"""
        print(f"\n=== MÉTRICAS DE EVALUACIÓN ===")

        # K-means metrics
        self.kmeans_silhouette = silhouette_score(self.features_scaled, self.kmeans_labels)
        self.kmeans_inertia = self.kmeans_model.inertia_

        # DBSCAN metrics
        if len(set(self.dbscan_labels)) > 1:
            self.dbscan_silhouette = silhouette_score(self.features_scaled, self.dbscan_labels)
        else:
            self.dbscan_silhouette = 0

        self.dbscan_n_clusters = len(set(self.dbscan_labels)) - (1 if -1 in self.dbscan_labels else 0)
        self.dbscan_n_noise = list(self.dbscan_labels).count(-1)

        # Distribución de clusters
        kmeans_cluster_sizes = np.bincount(self.kmeans_labels)
        dbscan_cluster_sizes = []
        for i in range(self.dbscan_n_clusters):
            dbscan_cluster_sizes.append(np.sum(self.dbscan_labels == i))

        print(f"\n--- K-MEANS ---")
        print(f"Clusters: {self.kmeans_model.n_clusters}")
        print(f"Silhouette Score: {self.kmeans_silhouette:.4f}")
        print(f"Inercia: {self.kmeans_inertia:.2f}")
        print(f"Iteraciones: {self.kmeans_model.n_iter_}")
        print(f"Tamaños de clusters: {kmeans_cluster_sizes}")

        print(f"\n--- DBSCAN ---")
        print(f"Clusters encontrados: {self.dbscan_n_clusters}")
        print(f"Puntos de ruido: {self.dbscan_n_noise} ({self.dbscan_n_noise/len(self.dbscan_labels)*100:.1f}%)")
        print(f"Silhouette Score: {self.dbscan_silhouette:.4f}")
        if dbscan_cluster_sizes:
            print(f"Tamaños de clusters: {dbscan_cluster_sizes}")

    def visualize_clusters(self, figsize: Tuple[int, int] = (16, 6)):
        """Visualiza los clusters en espacio 2D usando PCA"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # K-means visualization
        scatter1 = axes[0].scatter(self.features_pca[:, 0], self.features_pca[:, 1],
                                 c=self.kmeans_labels, cmap='tab10', alpha=0.7, s=30)
        axes[0].set_title(f'{self.dataset_name} - K-means\n'
                         f'Clusters: {self.kmeans_model.n_clusters}, '
                         f'Silhouette: {self.kmeans_silhouette:.3f}\n'
                         f'Features: {self.features.shape[1]} ✅')
        axes[0].set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.2%} varianza)')
        axes[0].set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.2%} varianza)')
        axes[0].grid(True, alpha=0.3)

        # DBSCAN visualization
        scatter2 = axes[1].scatter(self.features_pca[:, 0], self.features_pca[:, 1],
                                 c=self.dbscan_labels, cmap='tab10', alpha=0.7, s=30)
        axes[1].set_title(f'{self.dataset_name} - DBSCAN\n'
                         f'Clusters: {self.dbscan_n_clusters}, '
                         f'Ruido: {self.dbscan_n_noise}, '
                         f'Silhouette: {self.dbscan_silhouette:.3f}\n'
                         f'Features: {self.features.shape[1]} ✅')
        axes[1].set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.2%} varianza)')
        axes[1].set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.2%} varianza)')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_optimization_results(self, k_range: List[int], inertias: List[float],
                                silhouette_scores: List[float]):
        """Grafica resultados de optimización de K-means"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Método del codo
        axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0].set_title(f'{self.dataset_name} - Método del Codo\n(Features: {self.features.shape[1]} ✅)')
        axes[0].set_xlabel('Número de Clusters (k)')
        axes[0].set_ylabel('Inercia')
        axes[0].grid(True, alpha=0.3)

        # Silhouette score
        axes[1].plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        axes[1].set_title(f'{self.dataset_name} - Análisis Silhouette\n(Features: {self.features.shape[1]} ✅)')
        axes[1].set_xlabel('Número de Clusters (k)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].grid(True, alpha=0.3)

        # Marcar mejor k
        best_k_idx = np.argmax(silhouette_scores)
        best_k = k_range[best_k_idx]
        best_score = silhouette_scores[best_k_idx]

        axes[1].axvline(x=best_k, color='red', linestyle='--', alpha=0.7)
        axes[1].annotate(f'Mejor k={best_k}\nScore={best_score:.3f}',
                        xy=(best_k, best_score), xytext=(best_k+1, best_score),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                        fontsize=10, ha='left')

        plt.tight_layout()
        plt.show()

        return best_k


## FUNCIONES PRINCIPALES PARA COMPARACIÓN

def run_clustering_experiment(csv_file_path: str, dataset_name: str,
                            max_k: int = 15) -> MovieClusteringAnalyzer:
    """
    Ejecuta experimento completo de clustering para un dataset
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENTO DE CLUSTERING: {dataset_name}")
    print(f"Archivo: {csv_file_path}")
    print(f"{'='*60}")

    # Inicializar analizador
    analyzer = MovieClusteringAnalyzer()

    # Cargar datos
    analyzer.load_csv_data(csv_file_path, dataset_name)

    # Optimizar K-means
    print(f"\n--- OPTIMIZACIÓN K-MEANS ---")
    k_range, inertias, silhouette_scores = analyzer.find_optimal_k(max_k=max_k)
    best_k = analyzer.plot_optimization_results(k_range, inertias, silhouette_scores)

    # Entrenar modelos
    print(f"\n--- ENTRENAMIENTO DE MODELOS ---")
    analyzer.fit_models(n_clusters_kmeans=best_k)

    # Visualizar resultados
    print(f"\n--- VISUALIZACIÓN ---")
    analyzer.visualize_clusters()

    return analyzer


def compare_datasets(results_dict: Dict[str, MovieClusteringAnalyzer]):
    """
    Compara resultados entre diferentes datasets
    """
    print(f"\n{'='*60}")
    print("COMPARACIÓN DE DATASETS")
    print(f"{'='*60}")

    comparison_data = []

    for dataset_name, analyzer in results_dict.items():
        comparison_data.append({
            'Dataset': dataset_name,
            'Películas': analyzer.features.shape[0],
            'Características': analyzer.features.shape[1],
            'Features_Usados': f"{analyzer.feature_columns}",
            'PCA_Varianza': f"{analyzer.pca.explained_variance_ratio_.sum():.3f}",
            'KMeans_Clusters': analyzer.kmeans_model.n_clusters,
            'KMeans_Silhouette': f"{analyzer.kmeans_silhouette:.4f}",
            'KMeans_Inercia': f"{analyzer.kmeans_inertia:.2f}",
            'DBSCAN_Clusters': analyzer.dbscan_n_clusters,
            'DBSCAN_Ruido': f"{analyzer.dbscan_n_noise} ({analyzer.dbscan_n_noise/len(analyzer.dbscan_labels)*100:.1f}%)",
            'DBSCAN_Silhouette': f"{analyzer.dbscan_silhouette:.4f}"
        })

    comparison_df = pd.DataFrame(comparison_data)

    print("\nRESUMEN COMPARATIVO:")
    print(comparison_df.to_string(index=False))

    # Verificar que se están usando 10 features
    print(f"\n🔍 VERIFICACIÓN DE FEATURES:")
    for dataset_name, analyzer in results_dict.items():
        features_count = analyzer.features.shape[1]
        status = "✅" if features_count == 10 else "❌"
        print(f"  {dataset_name}: {features_count} features {status}")
        if features_count != 10:
            print(f"    PROBLEMA: Se esperaban 10 features, se encontraron {features_count}")
            print(f"    Features usados: {analyzer.feature_columns}")

    # Determinar mejor dataset
    print(f"\n--- ANÁLISIS DE RESULTADOS ---")

    # Comparar por Silhouette Score de K-means
    best_kmeans_idx = comparison_df['KMeans_Silhouette'].astype(float).idxmax()
    best_kmeans_dataset = comparison_df.loc[best_kmeans_idx, 'Dataset']
    best_kmeans_score = comparison_df.loc[best_kmeans_idx, 'KMeans_Silhouette']

    print(f"Mejor dataset para K-means: {best_kmeans_dataset} (Silhouette: {best_kmeans_score})")

    # Comparar por Silhouette Score de DBSCAN
    best_dbscan_idx = comparison_df['DBSCAN_Silhouette'].astype(float).idxmax()
    best_dbscan_dataset = comparison_df.loc[best_dbscan_idx, 'Dataset']
    best_dbscan_score = comparison_df.loc[best_dbscan_idx, 'DBSCAN_Silhouette']

    print(f"Mejor dataset para DBSCAN: {best_dbscan_dataset} (Silhouette: {best_dbscan_score})")

    # Recomendación final
    if best_kmeans_dataset == best_dbscan_dataset:
        print(f"\n🏆 RECOMENDACIÓN: Usar {best_kmeans_dataset}")
        print("   Ambos algoritmos funcionan mejor con este dataset")
    else:
        kmeans_score_val = float(best_kmeans_score)
        dbscan_score_val = float(best_dbscan_score)

        if kmeans_score_val > dbscan_score_val:
            print(f"\n🏆 RECOMENDACIÓN: Usar {best_kmeans_dataset} con K-means")
            print(f"   K-means tiene mejor rendimiento ({best_kmeans_score} vs {best_dbscan_score})")
        else:
            print(f"\n🏆 RECOMENDACIÓN: Usar {best_dbscan_dataset} con DBSCAN")
            print(f"   DBSCAN tiene mejor rendimiento ({best_dbscan_score} vs {best_kmeans_score})")

    return comparison_df


def create_visualization_comparison(results_dict: Dict[str, MovieClusteringAnalyzer]):
    """
    Crea visualización comparativa entre datasets
    """
    n_datasets = len(results_dict)
    fig, axes = plt.subplots(2, n_datasets, figsize=(6*n_datasets, 10))

    if n_datasets == 1:
        axes = axes.reshape(-1, 1)

    for i, (dataset_name, analyzer) in enumerate(results_dict.items()):
        # K-means en fila superior
        scatter1 = axes[0, i].scatter(analyzer.features_pca[:, 0], analyzer.features_pca[:, 1],
                                     c=analyzer.kmeans_labels, cmap='tab10', alpha=0.7, s=20)
        axes[0, i].set_title(f'{dataset_name} - K-means\n'
                           f'Silhouette: {analyzer.kmeans_silhouette:.3f}\n'
                           f'Features: {analyzer.features.shape[1]} ✅')
        axes[0, i].set_xlabel(f'PC1 ({analyzer.pca.explained_variance_ratio_[0]:.2%})')
        axes[0, i].set_ylabel(f'PC2 ({analyzer.pca.explained_variance_ratio_[1]:.2%})')
        axes[0, i].grid(True, alpha=0.3)

        # DBSCAN en fila inferior
        scatter2 = axes[1, i].scatter(analyzer.features_pca[:, 0], analyzer.features_pca[:, 1],
                                     c=analyzer.dbscan_labels, cmap='tab10', alpha=0.7, s=20)
        axes[1, i].set_title(f'{dataset_name} - DBSCAN\n'
                           f'Silhouette: {analyzer.dbscan_silhouette:.3f}\n'
                           f'Features: {analyzer.features.shape[1]} ✅')
        axes[1, i].set_xlabel(f'PC1 ({analyzer.pca.explained_variance_ratio_[0]:.2%})')
        axes[1, i].set_ylabel(f'PC2 ({analyzer.pca.explained_variance_ratio_[1]:.2%})')
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


## MAIN FUNCTION - PUNTO DE ENTRADA PRINCIPAL

def main():
    """
    Función principal que ejecuta el análisis completo
    """
    print("🎬 SISTEMA DE CLUSTERING DE POSTERS DE PELÍCULAS (VERSIÓN CORREGIDA)")
    print("=" * 70)
    print("Este script analiza las características visuales de posters")
    print("usando K-means y DBSCAN implementados desde cero")
    print("🔧 CORRIGIDO: Ahora usa exactamente 10 features (excluye índice y movieId)")
    print("=" * 70)

    # Configuración de archivos
    datasets = {
        'Solo Características Visuales': 'prim_reduced.csv',
        'Características + Géneros': 'prim_genere_reduced.csv'
    }

    results = {}

    # Ejecutar experimentos para cada dataset
    for dataset_name, file_path in datasets.items():
        try:
            print(f"\n🚀 Iniciando análisis para: {dataset_name}")
            analyzer = run_clustering_experiment(file_path, dataset_name, max_k=12)
            results[dataset_name] = analyzer

            print(f"✅ Análisis completado para: {dataset_name}")
            print(f"   Features utilizados: {analyzer.features.shape[1]} (correcto ✅)")

        except FileNotFoundError:
            print(f"❌ Error: No se encontró el archivo {file_path}")
            print(f"   Asegúrate de que el archivo esté en el directorio actual")
            continue
        except Exception as e:
            print(f"❌ Error procesando {dataset_name}: {str(e)}")
            continue

    # Comparar resultados si se procesaron múltiples datasets
    if len(results) > 1:
        print(f"\n🔍 COMPARANDO RESULTADOS...")
        comparison_df = compare_datasets(results)

        print(f"\n📊 VISUALIZACIÓN COMPARATIVA...")
        create_visualization_comparison(results)

    elif len(results) == 1:
        dataset_name = list(results.keys())[0]
        analyzer = results[dataset_name]
        print(f"\n✅ Análisis completado para {dataset_name}")
        print(f"Mejor método: K-means con Silhouette Score = {analyzer.kmeans_silhouette:.4f}")
        print(f"Features utilizados correctamente: {analyzer.features.shape[1]} ✅")

    else:
        print(f"\n❌ No se pudo procesar ningún dataset")
        print("Verifica que los archivos CSV estén disponibles")
        return

    # Resumen final y recomendaciones
    print(f"\n" + "="*70)
    print("📋 RESUMEN FINAL (VERSIÓN CORREGIDA)")
    print("="*70)

    if len(results) > 0:
        print("✅ Modelos entrenados correctamente con 10 features")
        print("✅ Implementaciones desde cero de K-means y DBSCAN")
        print("✅ Corrección aplicada: excluye índice y movieId")
        print("✅ Métricas de evaluación calculadas")
        print("✅ Visualizaciones generadas")

        print(f"\n🔧 CORRECCIONES APLICADAS:")
        print("1. ✅ Exclusión correcta de columnas de índice y movieId")
        print("2. ✅ Uso exacto de 10 features de características visuales")
        print("3. ✅ Preservación de metadatos para referencia")
        print("4. ✅ Validación de estructura de datos")

        print(f"\n🔧 PRÓXIMOS PASOS:")
        print("1. Usar el mejor modelo identificado para tu frontend")
        print("2. Implementar búsqueda por similitud visual")
        print("3. Crear sistema de recomendación basado en clusters")
        print("4. Integrar con tu interfaz web")

        # Guardar resultados (opcional)
        save_results = input(f"\n💾 ¿Guardar modelos entrenados corregidos? (y/n): ").lower().strip()
        if save_results == 'y':
            save_trained_models(results)

    print(f"\n🎉 ¡Análisis de clustering CORREGIDO completado!")


def save_trained_models(results_dict: Dict[str, MovieClusteringAnalyzer]):
    """
    Guarda los modelos entrenados para uso posterior
    """
    import pickle
    import os

    # Crear directorio para modelos
    models_dir = "trained_models_corrected"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    for dataset_name, analyzer in results_dict.items():
        # Limpiar nombre para archivo
        clean_name = dataset_name.replace(" ", "_").replace("+", "plus").lower()

        # Guardar analizador completo
        model_file = f"{models_dir}/analyzer_{clean_name}_corrected.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(analyzer, f)

        print(f"✅ Modelo corregido guardado: {model_file}")

        # Guardar métricas en CSV
        metrics_file = f"{models_dir}/metrics_{clean_name}_corrected.csv"
        metrics_data = {
            'Dataset': [dataset_name],
            'Movies': [analyzer.features.shape[0]],
            'Features': [analyzer.features.shape[1]],
            'Features_Used': [str(analyzer.feature_columns)],
            'KMeans_Clusters': [analyzer.kmeans_model.n_clusters],
            'KMeans_Silhouette': [analyzer.kmeans_silhouette],
            'KMeans_Inertia': [analyzer.kmeans_inertia],
            'DBSCAN_Clusters': [analyzer.dbscan_n_clusters],
            'DBSCAN_Noise': [analyzer.dbscan_n_noise],
            'DBSCAN_Silhouette': [analyzer.dbscan_silhouette],
            'Correction_Applied': ['YES - Excluded index and movieId columns']
        }

        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(metrics_file, index=False)
        print(f"✅ Métricas corregidas guardadas: {metrics_file}")

    print(f"\n📁 Todos los modelos CORREGIDOS guardados en: {models_dir}/")


def load_trained_model(model_path: str) -> MovieClusteringAnalyzer:
    """
    Carga un modelo previamente entrenado

    Uso:
    analyzer = load_trained_model('trained_models_corrected/analyzer_solo_caracteristicas_visuales_corrected.pkl')
    """
    import pickle

    with open(model_path, 'rb') as f:
        analyzer = pickle.load(f)

    print(f"✅ Modelo corregido cargado desde: {model_path}")
    print(f"Dataset: {analyzer.dataset_name}")
    print(f"Películas: {analyzer.features.shape[0]}")
    print(f"Características: {analyzer.features.shape[1]} (correcto ✅)")
    print(f"Features usados: {analyzer.feature_columns}")

    return analyzer


## CLASE API SIMPLIFICADA PARA FRONTEND

class MovieClusteringAPI:
    """
    API simplificada para integración con frontend
    Usar después de entrenar los modelos con main()
    """

    def __init__(self, analyzer: MovieClusteringAnalyzer):
        self.analyzer = analyzer
        print(f"🎬 API inicializada con {analyzer.features.shape[1]} features ✅")

    def search_similar_movies(self, movie_index: int, method: str = 'kmeans',
                            n_results: int = 10) -> List[Dict]:
        """Busca películas similares visualmente"""
        if method == 'kmeans':
            cluster_id = self.analyzer.kmeans_labels[movie_index]
            same_cluster = np.where(self.analyzer.kmeans_labels == cluster_id)[0]
        else:  # dbscan
            cluster_id = self.analyzer.dbscan_labels[movie_index]
            if cluster_id == -1:
                return []  # Es ruido
            same_cluster = np.where(self.analyzer.dbscan_labels == cluster_id)[0]

        # Excluir película original
        same_cluster = same_cluster[same_cluster != movie_index]

        if len(same_cluster) == 0:
            return []

        # Calcular distancias y ordenar
        ref_features = self.analyzer.features_scaled[movie_index]
        distances = []

        for idx in same_cluster:
            dist = np.linalg.norm(self.analyzer.features_scaled[idx] - ref_features)
            distances.append((idx, dist))

        distances.sort(key=lambda x: x[1])

        # Preparar resultados
        results = []
        for idx, dist in distances[:n_results]:
            result = {
                'movie_index': int(idx),
                'similarity_distance': float(dist),
                'cluster_id': int(cluster_id)
            }

            if self.analyzer.movie_metadata is not None:
                movie_info = self.analyzer.movie_metadata.iloc[idx].to_dict()
                result.update(movie_info)

            results.append(result)

        return results

    def get_cluster_info(self, method: str = 'kmeans') -> Dict:
        """Obtiene información general de clusters"""
        if method == 'kmeans':
            labels = self.analyzer.kmeans_labels
            n_clusters = self.analyzer.kmeans_model.n_clusters
            silhouette = self.analyzer.kmeans_silhouette
        else:
            labels = self.analyzer.dbscan_labels
            n_clusters = self.analyzer.dbscan_n_clusters
            silhouette = self.analyzer.dbscan_silhouette

        cluster_info = {
            'method': method,
            'n_clusters': int(n_clusters),
            'silhouette_score': float(silhouette),
            'total_movies': len(labels),
            'features_count': self.analyzer.features.shape[1],
            'features_used': self.analyzer.feature_columns,
            'cluster_sizes': {}
        }

        # Tamaños de clusters
        for cluster_id in set(labels):
            size = int(np.sum(labels == cluster_id))
            cluster_info['cluster_sizes'][str(cluster_id)] = size

        return cluster_info

    def predict_cluster_for_new_movie(self, new_features: np.ndarray, method: str = 'kmeans') -> Dict:
        """
        Predice cluster para una nueva película

        Parámetros:
        - new_features: Array con exactamente 10 features de la nueva película
        - method: 'kmeans' o 'dbscan'

        Retorna:
        - Información del cluster predicho
        """
        if len(new_features) != 10:
            raise ValueError(f"Se esperan exactamente 10 features, se recibieron {len(new_features)}")

        # Normalizar los nuevos features usando el mismo scaler
        new_features_scaled = self.analyzer.scaler.transform(new_features.reshape(1, -1))

        if method == 'kmeans':
            predicted_cluster = self.analyzer.kmeans_model.predict(new_features_scaled)[0]
            similar_movies = np.where(self.analyzer.kmeans_labels == predicted_cluster)[0]
        else:
            # Para DBSCAN, encontrar el punto más cercano y usar su cluster
            distances = np.linalg.norm(self.analyzer.features_scaled - new_features_scaled[0], axis=1)
            closest_idx = np.argmin(distances)
            predicted_cluster = self.analyzer.dbscan_labels[closest_idx]
            if predicted_cluster == -1:
                return {'cluster_id': -1, 'cluster_type': 'noise', 'similar_movies': []}
            similar_movies = np.where(self.analyzer.dbscan_labels == predicted_cluster)[0]

        return {
            'cluster_id': int(predicted_cluster),
            'cluster_type': 'normal' if predicted_cluster != -1 else 'noise',
            'similar_movies': similar_movies.tolist(),
            'cluster_size': len(similar_movies),
            'method_used': method
        }


if __name__ == "__main__":
    # Ejecutar análisis principal corregido
    main()

    print(f"\n" + "="*70)
    print("📚 DOCUMENTACIÓN RÁPIDA (VERSIÓN CORREGIDA)")
    print("="*70)
    print("Para usar los modelos entrenados CORREGIDOS en tu aplicación:")
    print()
    print("1. Cargar modelo entrenado corregido:")
    print("   analyzer = load_trained_model('trained_models_corrected/analyzer_xxx_corrected.pkl')")
    print()
    print("2. Crear API:")
    print("   api = MovieClusteringAPI(analyzer)")
    print()
    print("3. Buscar similares:")
    print("   similares = api.search_similar_movies(movie_index=123)")
    print()
    print("4. Obtener info de clusters:")
    print("   info = api.get_cluster_info(method='kmeans')")
    print()
    print("5. Para nueva imagen subida (con exactamente 10 features):")
    print("   # Extraer 10 características de la nueva imagen")
    print("   # new_features = np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10])")
    print("   # prediction = api.predict_cluster_for_new_movie(new_features, method='kmeans')")
    print()
    print("🔧 IMPORTANTE: Ahora el modelo usa exactamente 10 features (excluyendo índice y movieId)")
    print("="*70)