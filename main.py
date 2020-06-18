import pygame
import random
import os
import time
import math
import numpy as np
from copy import deepcopy
from math import sin, radians, degrees, copysign
from pygame.math import Vector2
from shapely.geometry import LineString, Point

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def predict(inputs, theta_1, theta_2):
	z2 = np.dot(inputs, theta_1.T)
	a2 = sigmoid(z2)

	z3 = np.dot(a2, theta_2.T)

	index = 0
	results = []
	for i in range (len(z3)):
		if (z3[i] > 0):
			results.append(i)
		index += 1

	return (results)

def neuralNetwork(data, weights):
	theta_1 = weights[0]
	theta_2 = weights[1]

	index = predict(data, theta_1, theta_2)
	moves = []
	for i in range(len(index)):
		if (index[i] == 0):
			moves.append('up')
		elif (index[i] == 1):
			moves.append('down')
		elif (index[i] == 2):
			moves.append('left')
		elif (index[i] == 3):
			moves.append('right')
	return (moves)


class Car:
	def __init__(self, x, y, car_img, angle=160.0, max_steering=50, max_acceleration = 5.0):
		self.position = Vector2(x, y)
		self.image = car_img
		self.rect = self.image.get_rect(center=(x, y))
		self.size = Vector2(40, 20)
		self.acceleration = 0
		self.velocity = Vector2(0.0, 0.0)
		self.angle = angle
		self.steering = 0.0
		self.orig_image = car_img
		self.max_steering = max_steering
		self.max_velocity = 20
		self.max_acceleration = max_acceleration
		self.mask = pygame.mask.from_surface(self.image, 50)
		self.alive = True

	def update(self):
		self.velocity += (self.acceleration, 0)
		angular_velocity = self.steering * 0.001 * self.velocity.x
		self.position += self.velocity.rotate(-self.angle)
		self.rect.center = self.position
		self.angle += degrees(angular_velocity)
		self.image = pygame.transform.rotate(self.orig_image, self.angle)
		self.rect = self.image.get_rect(center=self.rect.center)
		self.mask = pygame.mask.from_surface(self.image, 50)


class Game:
	def __init__(self):
		self.width = 960
		self.height = 540
		self.screen = pygame.display.set_mode((self.width, self.height))
		self.tracks_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs","tracks1.png")).convert_alpha(), (self.width, self.height))
		self.car_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs", "car.png")).convert_alpha(), (40, 20))
		self.gates = self.initGates()

	def draw(self, cars):
		self.screen.fill((255, 255, 255))
		for gate in self.gates:
			pygame.draw.line(self.screen, (255, 0, 0), gate[0],gate[1])
		self.screen.blit(self.tracks_img,(0, 0))
		for car in cars:
			if (car.alive):
				self.screen.blit(car.image, car.rect)
		pygame.display.flip()

	def initGates(self):
		gates = []
		gates.extend(([[200, 400], [200, 540]], [[150, 400], [110, 550]], [[50, 450], [130, 390]], [[30, 360], [130, 350]], [[25, 290], [125, 290]], [[45, 140], [160, 160]], [[220, 30], [220, 130]],
		[[370, 60], [320, 140]], [[360, 230], [440, 170]], [[520, 210], [520, 310]], [[600, 150], [690, 190]], [[740, 25], [740, 125]], [[880, 70], [810, 140]], [[840, 220], [940, 220]],
		[[800, 340], [900, 340]], [[780, 400], [860, 500]], [[700, 400], [700, 540]], [[600, 400], [600, 540]], [[500, 400], [500, 540]], [[400, 400], [400, 540]], [[300, 400], [300, 540]]))

		return (gates)

class Ai:
	def __init__(self, layers):
		self.layers = layers
		self.weights = self.initWeights()
		self.stepSinceLastGate = 0
		self.steps = 0
		self.gatesPassed = 0
		self.fitness = 0
		self.traveled = 0

	def initWeights(self):
		theta_1 = np.random.uniform(-1, 1, [self.layers[1], self.layers[0]])
		theta_2 = np.random.uniform(-1, 1, [self.layers[2], self.layers[1]])
		weights = []
		weights.append(theta_1)
		weights.append(theta_2)
		return (weights)

	def mutate(self):
		for i in range(len(self.weights)):
			for j in range(len(self.weights[i])):
				for k in range(len(self.weights[i][j])):
					if (random.randint(1, 100) <=  5):
						x = self.weights[i][j][k] + random.uniform(-1, 1)
						x = max(min(x, 1), -1)
						self.weights[i][j] =  x
	
	def reset(self):
		self.stepSinceLastGate = 0
		self.steps = 0
		self.gatesPassed = 0
		self.fitness = 0
		self.traveled = 0

def run(ais):
	game = Game()
	cars = []
	for i in range(len(ais)):
		cars.append(Car(230, 460, game.car_img))

	numberAlive = len(ais)

	# fps = 1
	# clock = pygame.time.Clock()

	while (numberAlive > 0):
		for i in range(len(cars)):
			car = cars[i]
			if (car.alive):
				game.draw(cars)

				# clock.tick(fps)
				
				# Event queue
				for event in pygame.event.get():
					if event.type == pygame.QUIT:
						quit()

				if (ais[i].stepSinceLastGate >= 150):
					car.alive = False

				cos = math.cos(math.radians(car.angle))
				sin = math.sin(math.radians(car.angle))

				directions = [
				[[cos * (car.size[0] / 2), -sin * (car.size[0] / 2)], cos, -sin],
				[[sin * (car.size[1] / 2), cos * (car.size[1] / 2)], sin, cos],
				[[-cos * (car.size[0] / 2), sin * (car.size[0] / 2)], -cos, sin],
				[[-sin * (car.size[1] / 2), -cos * (car.size[1] / 2)], -sin, -cos],
				[[(cos * (car.size[0] / 2)) - (sin * (car.size[1] / 2)), (-sin * (car.size[0] / 2)) - (cos * (car.size[1] / 2))], cos - sin, -sin - cos],
				[[(cos * (car.size[0] / 2)) + (sin * (car.size[1] / 2)), (-sin * (car.size[0] / 2)) + (cos * (car.size[1] / 2))], cos + sin, -sin + cos],
				[[(-cos * (car.size[0] / 2)) + (sin * (car.size[1] / 2)), (sin * (car.size[0] / 2)) + (cos * (car.size[1] / 2))], -cos + sin, sin + cos],
				[[(-cos * (car.size[0] / 2)) - (sin * (car.size[1] / 2)), (sin * (car.size[0] / 2)) - (cos * (car.size[1] / 2))], -cos - sin, sin - cos]]
				
				distances = []

				for direction in directions:
					inital = direction[0] + Vector2(car.rect.center)
					position = inital
					gateDistance = 0
					dead = True
					while (game.screen.get_at([int(position[0]), int(position[1])]) != (0, 0, 0, 255)):
						dead = False
						position = [position[0] + direction[1], position[1] + direction[2]]
					if (dead):
						car.alive = False
						numberAlive -= 1
						ais[i].fitness = ais[i].traveled + (ais[i].gatesPassed**2.1 * 500)
						break
					distance = max(abs(inital[0] - position[0]), abs(inital[1] - position[1]))
					distance = 1 / distance if distance != 0 else 0
					distances.append(distance)
					line1 = LineString([inital, position])
					line2 = LineString(game.gates[ais[i].gatesPassed % len(game.gates)])
					point1 = line1.intersection(line2)
					if (point1.geom_type == 'Point'):
						point = Point(point1)
						distances.append(1 / max(abs(point.x - inital[0]), abs(point.y - inital[1])))
					else:
						distances.append( 1/ math.inf)
				
				if (car.alive):
					line1 = LineString([[(cos * (car.size[0] / 2)) - (sin * (car.size[1] / 2)), (-sin * (car.size[0] / 2)) - (cos * (car.size[1] / 2))] + Vector2(car.rect.center), [(-cos * (car.size[0] / 2)) + (sin * (car.size[1] / 2)), (sin * (car.size[0] / 2)) + (cos * (car.size[1] / 2))] + Vector2(car.rect.center)])
					line2 = LineString(game.gates[ais[i].gatesPassed % len(game.gates)])
					point1 = line1.intersection(line2)

					line3 = LineString([[(cos * (car.size[0] / 2)) + (sin * (car.size[1] / 2)), (-sin * (car.size[0] / 2)) + (cos * (car.size[1] / 2))] + Vector2(car.rect.center), [(-cos * (car.size[0] / 2)) - (sin * (car.size[1] / 2)), (sin * (car.size[0] / 2)) - (cos * (car.size[1] / 2))] + Vector2(car.rect.center)])
					line4 = LineString(game.gates[ais[i].gatesPassed % len(game.gates)])
					point2 = line3.intersection(line4)

					if (point1.geom_type == 'Point' or point2.geom_type == 'Point'):
						ais[i].gatesPassed += 1
						ais[i].stepSinceLastGate = 0

					inputs = distances
					inputs.extend((1 / car.velocity.x if car.velocity.x != 0 else 0, 1 / car.acceleration if car.acceleration != 0 else 0, 1 / car.steering if car.steering != 0 else 0, 1 / car.angle if car.angle != 0 else 0))

					if (len(inputs) < 12):
						car.alive = False
					
					commands = neuralNetwork(inputs, ais[i].weights)

					for command in commands:
						if (command == 'up'):
							car.acceleration += 1
						if (command == 'down'):
							car.acceleration -= 1
						car.acceleration = max(car.max_acceleration, min(car.acceleration, car.max_acceleration))
						if (command == 'right'):
							car.steering -= 10
						if (command == 'left'):
							car.steering += 10
						car.steering = max(-car.max_steering, min(car.steering, car.max_steering))

					# Logic
					car.update()

					ais[i].steps += 1
					ais[i].stepSinceLastGate += 1
					ais[i].traveled += car.velocity.x
				

def selectParent(ais, fitnessSum):
	rand = random.uniform(0, fitnessSum)
	currentSum = 0
	for ai in ais:
		currentSum += ai.fitness
		if (currentSum > rand):
			return (ai)

def getFitnessSum(ais):
	fitnessSum = 0
	for ai in ais:
		fitnessSum += ai.fitness
	return fitnessSum

def mixWeights(a, b):
	weights = deepcopy(a)
	for i in range(len(weights)):
		for j in range(len(weights[i])):
			for k in range(len(weights[i][j])):
				weights[i][j][k] = (a[i][j][k] if random.randint(0, 1) == 0 else b[i][j][k])
	return weights

if __name__ == '__main__':

	stop = False
	ais = []
	for _ in range(10):
		ais.append(Ai([20, 8, 4]))

	generation = 1
	
	while (not stop):
		for ai in ais:
			ai.reset()

		run(ais)

		fitnessSum = getFitnessSum(ais)
		children = []
		for _ in range(len(ais) - 1):
			parentA = selectParent(ais, fitnessSum)
			parentB = selectParent(ais, fitnessSum)
			child = deepcopy(parentA)
			child.weights =  mixWeights(parentA.weights, parentB.weights)
			if (random.randint(1, 100) <= 80):
				child.mutate()
			children.append(child)
		ais.sort(key=lambda x: x.fitness, reverse=True)
		print('Gen: {}, Fit: {}, Gates: {}, Traveled: {}'.format(generation, ais[0].fitness, ais[0].gatesPassed, ais[0].traveled))
		ais = [ais[0]] + children
		generation += 1
