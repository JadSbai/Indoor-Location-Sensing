import random

import math
import numpy as np
from typing import List, Tuple, Dict
from scipy.stats import gaussian_kde
from models.helpers import Map


class Particle:
    def __init__(self, position: Tuple[float, float], speed: float, heading: float, weight: float = 1.0):
        self.position = position
        self.speed = speed
        self.heading = heading
        self.weight = weight


class Cluster:
    def __init__(self, position, probability):
        self.position = position
        self.probability = probability


class ParticleFilter:
    particles: List[Particle]
    n_particles: int
    mean_speed: float
    resampling_threshold: float

    def __init__(self, n_particles: int, mean_speed: float, layout: Map, resampling_threshold: float = 0.5):
        self.n_particles = n_particles
        self.mean_speed = mean_speed
        self.resampling_threshold = resampling_threshold
        self.particles = []
        self.map = layout
        self.delta_t = 1
        self.var_speed = 0.01
        self.var_heading = np.pi/2

    def set_delta_t(self, new_delta_t):
        self.delta_t = new_delta_t

    def initialize_particles(self) -> None:
        self.particles = []
        while len(self.particles) < self.n_particles:
            x = random.randrange(0, self.map.width + 1)
            y = random.randrange(0, self.map.height + 1)
            if self.map.is_valid_position(x, y):
                speed = self.mean_speed
                heading = np.random.uniform(0, 2 * np.pi)
                weight = 1.0 / self.n_particles
                self.particles.append(Particle((x, y), speed, heading, weight))

    def apply_motion_model(self, particle: Particle) -> Tuple[float, float]:
        dx = particle.speed * np.cos(particle.heading) * self.delta_t
        dy = particle.speed * np.sin(particle.heading) * self.delta_t
        heading_diff = np.random.normal(0, self.var_heading)
        speed_diff = np.random.normal(0, self.var_speed)
        new_heading = particle.heading + heading_diff
        new_speed = particle.speed + speed_diff
        new_x = particle.position[0] + dx
        new_y = particle.position[1] + dy
        particle.position = (new_x, new_y)
        particle.heading = new_heading
        particle.speed = new_speed
        return particle.position

    def predict(self) -> None:
        for i in range(len(self.particles)):
            current_particle = self.particles[i]
            new_position = self.apply_motion_model(current_particle)
            if not self.map.is_valid_position(new_position[0], new_position[1]):
                current_particle.weight = 0

    def get_min_max_dist(self, cluster_position: Tuple[float, float]):
        max_dist = float('-inf')
        min_dist = float('inf')
        for p in self.particles:
            distance = math.dist(cluster_position, p.position)
            if distance > max_dist:
                max_dist = distance
            if distance < min_dist:
                min_dist = distance
        return min_dist, max_dist

    def compute_cluster_score(self, particle: Particle, cluster: Cluster):
        cluster_min_dist, cluster_max_dist = self.get_min_max_dist(cluster.position)
        cluster_particle_dist = math.dist(particle.position, cluster.position)
        distance_scaler = 1 - ((cluster_particle_dist - cluster_min_dist) / (cluster_max_dist - cluster_min_dist))
        return cluster.probability * distance_scaler

    def get_k_means_score(self, particle: Particle, clusters: List[Cluster]) -> float:
        score = 0
        for cluster in clusters:
            cluster_score = self.compute_cluster_score(particle, cluster)
            score += cluster_score
        return score

    def get_optics_score(self, particle: Particle, core_points, kde, prob_distribution):
        # Find the nearest core point and cluster for the particle
        total = 0
        for cluster_label in core_points:
            if cluster_label != -1:
                total_x = 0
                total_y = 0
                for point in core_points[cluster_label]:
                    total_x += point[0]
                    total_y += point[1]
                centroid = (total_x / len(core_points[cluster_label]), total_y / len(core_points[cluster_label]))
                total += self.compute_cluster_score(particle, Cluster(centroid, prob_distribution[cluster_label]))
        weight = total
        density = kde(particle.position)[0]
        weight *= 5 * density
        return weight

    def update_weights(self, prob_distribution: List[float],
                       cluster_centroids: List[Tuple[float, float]] = None,
                       core_points: List[Tuple[float, float]] = None,
                       label_to_points: Dict[int, List[Tuple[float, float]]] = None) -> None:
        is_kMeans = cluster_centroids is not None
        clusters = []
        kde = None
        if is_kMeans:
            for label in range(len(prob_distribution)):
                position = cluster_centroids[label]
                probability = prob_distribution[label]
                clusters.append(Cluster(position, probability))
        else:
            # Compute the kernel density estimation using all core points
            all_core_points = np.vstack(core_points)
            kde = gaussian_kde(all_core_points.T)

        weight_sum = 0
        # Update the weight of each particle
        for i, particle in enumerate(self.particles):
            if particle.weight != 0:
                new_weight = self.get_k_means_score(particle, clusters) if is_kMeans else self.get_optics_score(
                    particle, label_to_points, kde, prob_distribution)
                particle.weight = new_weight
                weight_sum += new_weight

        # Normalize weights
        for particle in self.particles:
            particle.weight = particle.weight / weight_sum

        if self.needs_resampling():
            self.resample()

    def needs_resampling(self) -> bool:
        sum_weights_squared = sum([particle.weight ** 2 for particle in self.particles])
        neff = 1 / sum_weights_squared
        return neff / self.n_particles < self.resampling_threshold

    def resample(self) -> None:
        new_particles = []
        particles = self.multinomial_resample()

        for p in particles:
            new_particles.append(Particle(p.position, p.speed, p.heading, 1.0))

        self.particles = new_particles

    def multinomial_resample(self) -> List[Particle]:
        num_particles = len(self.particles)

        # Calculate the cumulative weights
        cumulative_weights = []
        current_sum = 0
        for particle in self.particles:
            current_sum += particle.weight
            cumulative_weights.append(current_sum)

        # Resample particles
        resampled_particles = []
        for _ in range(num_particles):
            r = random.uniform(0, cumulative_weights[-1])
            idx = 0
            while r > cumulative_weights[idx]:
                idx += 1
            resampled_particles.append(self.particles[idx])

        return resampled_particles

    def systematic_resampling(self):
        n = len(self.particles)
        positions = [(i + random.uniform(0, 1)) / n for i in range(n)]

        resampled_particles = []
        cumulative_sum = 0.0
        j = 0

        for i in range(n):
            cumulative_sum += self.particles[i].weight
            while positions[j] < cumulative_sum:
                resampled_particles.append(self.particles[i])
                j += 1
                if j == n:
                    break

        return resampled_particles

    def average_estimate_position(self):
        particles = self.particles
        particles.sort(key=lambda p: p.weight, reverse=True)
        x = 0
        y = 0
        i = 0
        n = 0.1 * self.n_particles
        while i < n:
            x += particles[i].position[0]
            y += particles[i].position[1]
            i += 1
        return x / n, y / n

    def simple_estimate_position(self) -> Tuple[float, float]:
        particles = self.particles
        particles.sort(key=lambda x: x.weight, reverse=True)
        return particles[0].position

    def weighted_estimate_position(self):
        final_x = 0
        final_y = 0
        length = len(self.particles)
        for particle in self.particles:
            final_x += particle.position[0] * particle.weight
            final_y += particle.position[1] * particle.weight
        return final_x / length, final_y / length
