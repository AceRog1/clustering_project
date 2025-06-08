# K-means Clustering Script - Solo Características Visuales de Posters
# Proyecto Educativo de Machine Learning - Versión Simplificada y Corregida

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
import time
from typing import List, Tuple, Optional, Dict, Any

warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

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


class VisualClusteringAnalyzer:
    """Analizador de clustering para características visuales de posters - Solo K-means"""

    def __init__(self):
        self.features = None
        self.movie_metadata = None
        self.kmeans_model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.features_scaled = None
        self.features_pca = None
        self.feature_columns = None
        self.kmeans_labels = None
        self.kmeans_silhouette = None
        self.kmeans_inertia = None

    def load_visual_features(self, csv_file_path: str = 'prim_reduced.csv'):
        """
        Carga características visuales desde archivo CSV

        Parámetros:
        - csv_file_path: Ruta al archivo CSV (por defecto 'prim_reduced.csv')
        """
        print(f"🎬 Cargando características visuales desde: {csv_file_path}")

        # Leer CSV
        df = pd.read_csv(csv_file_path)
        print(f"Archivo cargado: {df.shape[0]} películas, {df.shape[1]} columnas")
        print(f"Columnas encontradas: {list(df.columns)}")

        # 🔧 CORRECCIÓN: Identificar exactamente las 10 características visuales
        excluded_columns = ['movieId']

        # La primera columna es el índice sin nombre
        first_col = df.columns[0]
        if first_col == '' or 'Unnamed' in str(first_col) or first_col.lower() in ['index', 'idx']:
            excluded_columns.append(first_col)

        # Las características visuales son las columnas '0' a '9'
        expected_features = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        feature_columns = [col for col in expected_features if col in df.columns]

        self.feature_columns = feature_columns

        print(f"\n✅ ESTRUCTURA IDENTIFICADA:")
        print(f"  - Columnas excluidas: {excluded_columns}")
        print(f"  - Características visuales: {feature_columns}")
        print(f"  - Número de features: {len(feature_columns)}")

        if len(feature_columns) != 10:
            raise ValueError(f"❌ Error: Se esperaban 10 características visuales, se encontraron {len(feature_columns)}")

        # Extraer solo las características visuales
        self.features = df[feature_columns].values

        # Crear metadatos
        metadata_dict = {
            'movie_index': range(len(df)),
            'movieId': df['movieId'].values if 'movieId' in df.columns else range(len(df))
        }

        # Agregar índice original si existe
        if first_col in df.columns and first_col in excluded_columns and first_col != 'movieId':
            metadata_dict['original_index'] = df[first_col].values

        self.movie_metadata = pd.DataFrame(metadata_dict)

        # Verificar calidad de los datos
        if np.any(np.isnan(self.features)) or np.any(np.isinf(self.features)):
            print("⚠️ Limpiando valores NaN/infinitos...")
            self.features = np.nan_to_num(self.features, nan=0.0, posinf=1e6, neginf=-1e6)

        # Normalizar características
        print("📊 Normalizando características...")
        self.features_scaled = self.scaler.fit_transform(self.features)

        # PCA para visualización
        print("🔍 Aplicando PCA para visualización...")
        self.features_pca = self.pca.fit_transform(self.features_scaled)

        print(f"\n=== RESUMEN DE DATOS ===")
        print(f"🎬 Películas analizadas: {self.features.shape[0]:,}")
        print(f"🎨 Características visuales: {self.features.shape[1]} ✅")
        print(f"📈 Varianza explicada por PCA: {self.pca.explained_variance_ratio_.sum():.1%}")

        # Estadísticas de características
        print(f"\n📊 Estadísticas de características visuales:")
        for i, col in enumerate(feature_columns):
            feature_data = self.features[:, i]
            print(f"  Feature {col}: μ={np.mean(feature_data):.3f}, "
                  f"σ={np.std(feature_data):.3f}, "
                  f"rango=[{np.min(feature_data):.3f}, {np.max(feature_data):.3f}]")

        return self

    def find_optimal_k(self, max_k: int = 15) -> Tuple[List[int], List[float], List[float]]:
        """Encuentra el número óptimo de clusters usando método del codo y silhouette"""
        k_range = range(2, max_k + 1)
        inertias = []
        silhouette_scores = []

        print(f"\n🔍 Buscando número óptimo de clusters (k=2 a {max_k})...")
        print("Esto puede tomar unos minutos...")

        for k in k_range:
            print(f"  📊 Evaluando k={k:2d}...", end=" ")
            start_time = time.time()

            # Entrenar K-means
            kmeans = KMeansCustom(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(self.features_scaled)

            # Calcular métricas
            inertias.append(kmeans.inertia_)
            sil_score = silhouette_score(self.features_scaled, labels)
            silhouette_scores.append(sil_score)

            elapsed = time.time() - start_time
            print(f"Silhouette: {sil_score:.4f} ({elapsed:.1f}s)")

        print(f"✅ Evaluación completada!")
        return list(k_range), inertias, silhouette_scores

    def plot_optimization_results(self, k_range: List[int], inertias: List[float],
                                silhouette_scores: List[float]) -> int:
        """Visualiza resultados de optimización y retorna el mejor k"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Método del codo
        axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0].set_title(f'Método del Codo\n({self.features.shape[1]} características visuales)')
        axes[0].set_xlabel('Número de Clusters (k)')
        axes[0].set_ylabel('Inercia (WCSS)')
        axes[0].grid(True, alpha=0.3)

        # Silhouette score
        axes[1].plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        axes[1].set_title(f'Análisis Silhouette\n({self.features.shape[1]} características visuales)')
        axes[1].set_xlabel('Número de Clusters (k)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].grid(True, alpha=0.3)

        # Identificar y marcar el mejor k
        best_k_idx = np.argmax(silhouette_scores)
        best_k = k_range[best_k_idx]
        best_score = silhouette_scores[best_k_idx]

        axes[1].axvline(x=best_k, color='red', linestyle='--', alpha=0.7)
        axes[1].annotate(f'Óptimo k={best_k}\nScore={best_score:.3f}',
                        xy=(best_k, best_score), xytext=(best_k+1, best_score+0.01),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                        fontsize=11, ha='left', fontweight='bold')

        plt.tight_layout()
        plt.show()

        print(f"\n🏆 RESULTADO: k óptimo = {best_k} (Silhouette Score = {best_score:.4f})")
        return best_k

    def train_kmeans(self, n_clusters: int):
        """Entrena el modelo K-means final"""
        if self.features_scaled is None:
            raise ValueError("❌ Primero carga los datos usando load_visual_features()")

        print(f"\n🤖 Entrenando K-means con {n_clusters} clusters...")
        start_time = time.time()

        self.kmeans_model = KMeansCustom(n_clusters=n_clusters, random_state=42)
        self.kmeans_labels = self.kmeans_model.fit_predict(self.features_scaled)

        training_time = time.time() - start_time

        # Calcular métricas
        self.kmeans_silhouette = silhouette_score(self.features_scaled, self.kmeans_labels)
        self.kmeans_inertia = self.kmeans_model.inertia_

        print(f"✅ Entrenamiento completado en {training_time:.2f} segundos")

        # Mostrar métricas finales
        self._show_final_metrics()

    def _show_final_metrics(self):
        """Muestra métricas finales del modelo"""
        print(f"\n=== MÉTRICAS FINALES ===")
        print(f"🎯 Clusters: {self.kmeans_model.n_clusters}")
        print(f"📊 Silhouette Score: {self.kmeans_silhouette:.4f}")
        print(f"📉 Inercia: {self.kmeans_inertia:.2f}")
        print(f"🔄 Iteraciones: {self.kmeans_model.n_iter_}")

        # Distribución de clusters
        cluster_sizes = np.bincount(self.kmeans_labels)
        print(f"📈 Distribución de clusters:")
        for i, size in enumerate(cluster_sizes):
            percentage = (size / len(self.kmeans_labels)) * 100
            print(f"   Cluster {i}: {size:,} películas ({percentage:.1f}%)")

    def visualize_clusters(self, figsize: Tuple[int, int] = (12, 8)):
        """Visualiza los clusters en espacio 2D usando PCA"""
        if self.kmeans_labels is None:
            raise ValueError("❌ Primero entrena el modelo usando train_kmeans()")

        plt.figure(figsize=figsize)

        # Crear scatter plot
        scatter = plt.scatter(self.features_pca[:, 0], self.features_pca[:, 1],
                            c=self.kmeans_labels, cmap='tab10', alpha=0.7, s=30)

        # Añadir centroides en espacio PCA
        centroids_pca = self.pca.transform(self.kmeans_model.centroids_)
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                   c='red', marker='x', s=200, linewidths=3, label='Centroides')

        plt.title(f'Clustering de Características Visuales de Posters\n'
                 f'K-means con {self.kmeans_model.n_clusters} clusters '
                 f'(Silhouette: {self.kmeans_silhouette:.3f})')
        plt.xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} de varianza)')
        plt.ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} de varianza)')
        plt.colorbar(scatter, label='Cluster ID')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def analyze_clusters(self):
        """Analiza las características de cada cluster"""
        if self.kmeans_labels is None:
            raise ValueError("❌ Primero entrena el modelo usando train_kmeans()")

        print(f"\n=== ANÁLISIS DETALLADO DE CLUSTERS ===")

        for cluster_id in range(self.kmeans_model.n_clusters):
            cluster_mask = self.kmeans_labels == cluster_id
            cluster_features = self.features[cluster_mask]
            cluster_size = np.sum(cluster_mask)

            print(f"\n🎨 CLUSTER {cluster_id} ({cluster_size:,} películas, {cluster_size/len(self.features)*100:.1f}%)")
            print(f"   Características promedio:")

            for i, feature_name in enumerate(self.feature_columns):
                feature_mean = np.mean(cluster_features[:, i])
                feature_std = np.std(cluster_features[:, i])
                global_mean = np.mean(self.features[:, i])

                # Indicar si está por encima o debajo del promedio global
                trend = "↑" if feature_mean > global_mean else "↓"
                print(f"     Feature {feature_name}: {feature_mean:.3f} ± {feature_std:.3f} {trend}")

    def get_cluster_examples(self, cluster_id: int, n_examples: int = 5) -> List[Dict]:
        """Obtiene ejemplos de películas de un cluster específico"""
        if self.kmeans_labels is None:
            raise ValueError("❌ Primero entrena el modelo usando train_kmeans()")

        cluster_mask = self.kmeans_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            return []

        # Seleccionar ejemplos aleatorios
        n_examples = min(n_examples, len(cluster_indices))
        example_indices = np.random.choice(cluster_indices, n_examples, replace=False)

        examples = []
        for idx in example_indices:
            example = {
                'movie_index': int(idx),
                'cluster_id': int(cluster_id),
                'movieId': self.movie_metadata.iloc[idx]['movieId'] if 'movieId' in self.movie_metadata.columns else None
            }
            examples.append(example)

        return examples


class MovieClusteringAPI:
    """API simplificada para integración con frontend - Solo K-means"""

    def __init__(self, analyzer: VisualClusteringAnalyzer):
        self.analyzer = analyzer
        if analyzer.kmeans_model is None:
            raise ValueError("❌ El analizador debe tener un modelo K-means entrenado")
        print(f"🎬 API inicializada: {analyzer.features.shape[1]} features, {analyzer.kmeans_model.n_clusters} clusters ✅")

    def search_similar_movies(self, movie_index: int, n_results: int = 10) -> List[Dict]:
        """Busca películas similares visualmente usando K-means"""
        cluster_id = self.analyzer.kmeans_labels[movie_index]
        same_cluster = np.where(self.analyzer.kmeans_labels == cluster_id)[0]

        # Excluir película original
        same_cluster = same_cluster[same_cluster != movie_index]

        if len(same_cluster) == 0:
            return []

        # Calcular distancias euclidianas en el espacio normalizado
        ref_features = self.analyzer.features_scaled[movie_index]
        distances = []

        for idx in same_cluster:
            dist = np.linalg.norm(self.analyzer.features_scaled[idx] - ref_features)
            distances.append((idx, dist))

        # Ordenar por distancia y tomar los más cercanos
        distances.sort(key=lambda x: x[1])

        results = []
        for idx, dist in distances[:n_results]:
            result = {
                'movie_index': int(idx),
                'similarity_distance': float(dist),
                'cluster_id': int(cluster_id),
                'movieId': self.analyzer.movie_metadata.iloc[idx]['movieId'] if 'movieId' in self.analyzer.movie_metadata.columns else None
            }
            results.append(result)

        return results

    def predict_cluster_for_new_movie(self, new_features: np.ndarray) -> Dict:
        """Predice cluster para una nueva película con 10 características visuales"""
        if len(new_features) != 10:
            raise ValueError(f"❌ Se esperan exactamente 10 features, se recibieron {len(new_features)}")

        # Normalizar usando el mismo scaler
        new_features_scaled = self.analyzer.scaler.transform(new_features.reshape(1, -1))

        # Predecir cluster
        predicted_cluster = self.analyzer.kmeans_model.predict(new_features_scaled)[0]

        # Obtener información del cluster
        cluster_mask = self.analyzer.kmeans_labels == predicted_cluster
        similar_movies = np.where(cluster_mask)[0]

        return {
            'predicted_cluster': int(predicted_cluster),
            'cluster_size': len(similar_movies),
            'similar_movies': similar_movies.tolist()[:20],  # Primeros 20
            'silhouette_score': float(self.analyzer.kmeans_silhouette)
        }

    def get_cluster_info(self) -> Dict:
        """Obtiene información general del clustering"""
        cluster_info = {
            'n_clusters': int(self.analyzer.kmeans_model.n_clusters),
            'total_movies': len(self.analyzer.kmeans_labels),
            'features_count': 10,
            'silhouette_score': float(self.analyzer.kmeans_silhouette),
            'inertia': float(self.analyzer.kmeans_inertia),
            'cluster_sizes': {}
        }

        # Tamaños de cada cluster
        for cluster_id in range(self.analyzer.kmeans_model.n_clusters):
            size = int(np.sum(self.analyzer.kmeans_labels == cluster_id))
            cluster_info['cluster_sizes'][str(cluster_id)] = size

        return cluster_info


def main():
    """Función principal - Análisis completo de clustering visual"""
    print("🎬 CLUSTERING DE CARACTERÍSTICAS VISUALES DE POSTERS")
    print("=" * 60)
    print("Análisis simplificado usando solo K-means")
    print("Dataset: Solo características visuales (10 features)")
    print("=" * 60)

    try:
        # 1. Inicializar analizador
        print(f"\n🚀 PASO 1: Inicializando analizador...")
        analyzer = VisualClusteringAnalyzer()

        # 2. Cargar datos
        print(f"\n📊 PASO 2: Cargando datos...")
        analyzer.load_visual_features('prim_reduced.csv')

        # 3. Encontrar k óptimo
        print(f"\n🔍 PASO 3: Optimizando número de clusters...")
        k_range, inertias, silhouette_scores = analyzer.find_optimal_k(max_k=12)
        best_k = analyzer.plot_optimization_results(k_range, inertias, silhouette_scores)

        # 4. Entrenar modelo final
        print(f"\n🤖 PASO 4: Entrenando modelo final...")
        analyzer.train_kmeans(n_clusters=best_k)

        # 5. Visualizar resultados
        print(f"\n📈 PASO 5: Visualizando clusters...")
        analyzer.visualize_clusters()

        # 6. Análisis detallado
        print(f"\n🔍 PASO 6: Análisis detallado de clusters...")
        analyzer.analyze_clusters()

        # 7. Crear API para uso en frontend
        print(f"\n🔧 PASO 7: Preparando API para frontend...")
        api = MovieClusteringAPI(analyzer)

        # Mostrar información de la API
        cluster_info = api.get_cluster_info()
        print(f"✅ API lista:")
        print(f"   - {cluster_info['total_movies']:,} películas")
        print(f"   - {cluster_info['n_clusters']} clusters")
        print(f"   - Silhouette Score: {cluster_info['silhouette_score']:.4f}")

        # 8. Guardar modelo
        save_model = input(f"\n💾 ¿Guardar modelo entrenado? (y/n): ").lower().strip()
        if save_model == 'y':
            save_trained_model(analyzer)

        print(f"\n🎉 ¡Análisis completado exitosamente!")
        print(f"\n📋 RESUMEN FINAL:")
        print(f"   ✅ Modelo K-means entrenado con {analyzer.features.shape[1]} características visuales")
        print(f"   ✅ {analyzer.kmeans_model.n_clusters} clusters identificados")
        print(f"   ✅ {analyzer.features.shape[0]:,} películas analizadas")
        print(f"   ✅ API lista para integración con frontend")

        return analyzer, api

    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo 'prim_reduced.csv'")
        print(f"   Asegúrate de que el archivo esté en el directorio actual")
        return None, None
    except Exception as e:
        print(f"❌ Error durante el análisis: {str(e)}")
        return None, None


def save_trained_model(analyzer: VisualClusteringAnalyzer):
    """Guarda el modelo entrenado"""
    import pickle
    import os

    models_dir = "visual_clustering_model"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Guardar analizador completo
    model_file = f"{models_dir}/visual_clustering_analyzer.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(analyzer, f)
    print(f"✅ Modelo guardado: {model_file}")

    # Guardar métricas
    metrics_file = f"{models_dir}/clustering_metrics.csv"
    metrics_data = {
        'total_movies': [analyzer.features.shape[0]],
        'features_count': [analyzer.features.shape[1]],
        'n_clusters': [analyzer.kmeans_model.n_clusters],
        'silhouette_score': [analyzer.kmeans_silhouette],
        'inertia': [analyzer.kmeans_inertia],
        'iterations': [analyzer.kmeans_model.n_iter_],
        'pca_variance_explained': [analyzer.pca.explained_variance_ratio_.sum()]
    }

    pd.DataFrame(metrics_data).to_csv(metrics_file, index=False)
    print(f"✅ Métricas guardadas: {metrics_file}")


def load_trained_model(model_path: str = "visual_clustering_model/visual_clustering_analyzer.pkl") -> VisualClusteringAnalyzer:
    """Carga un modelo previamente entrenado"""
    import pickle

    with open(model_path, 'rb') as f:
        analyzer = pickle.load(f)

    print(f"✅ Modelo cargado: {analyzer.features.shape[0]:,} películas, {analyzer.kmeans_model.n_clusters} clusters")
    return analyzer


if __name__ == "__main__":
    # Ejecutar análisis principal
    analyzer, api = main()

    if analyzer is not None:
        print(f"\n" + "="*60)
        print("📚 GUÍA DE USO RÁPIDA")
        print("="*60)
        print("Para usar el modelo en tu aplicación:")
        print()
        print("1. Cargar modelo:")
        print("   analyzer = load_trained_model()")
        print("   api = MovieClusteringAPI(analyzer)")
        print()
        print("2. Buscar películas similares:")
        print("   similares = api.search_similar_movies(movie_index=123, n_results=10)")
        print()
        print("3. Para nueva película (con 10 features visuales):")
        print("   features = np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10])")
        print("   resultado = api.predict_cluster_for_new_movie(features)")
        print()
        print("4. Información general:")
        print("   info = api.get_cluster_info()")
        print("="*60)