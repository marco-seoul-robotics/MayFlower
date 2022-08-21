import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pyproj
import csv
import json
import argparse


class FloodRegionDetector:
  def __init__(self, grid_resolution, epsg):
    self.data_list = []
    self.min_x, self.max_x = 1e9, -1e9
    self.min_y, self.max_y = 1e9, -1e9
    self.min_z, self.max_z = 1e9, -1e9
    self.res = grid_resolution
    self.epsg = epsg

  def parse_data(self):
    files = os.listdir('data')

    for file in files:
      if file[-3:] == "txt":
        f = open(os.path.join('data', file), 'r')
        while True:
          line = f.readline().strip()
          if not line: break
          parsed_line = line.split()
          x, y, z = float(parsed_line[0]), float(parsed_line[1]), float(parsed_line[2])
          self.data_list.append([x, y, z])
          if x < self.min_x:
            self.min_x = x
          if x > self.max_x:
            self.max_x = x
          if y < self.min_y:
            self.min_y = y
          if y > self.max_y:
            self.max_y = y
          if z < self.min_z:
            self.min_z = z
          if z > self.max_z:
            self.max_z = z
        f.close()

  def fill_grid(self):
    self.grid_size = np.array([int((self.max_x - self.min_x) / self.res) + 1, int((self.max_y - self.min_y) / self.res) + 1])
    self.grid = np.full(self.grid_size, 0, np.float)
    for data in self.data_list:
      row = int((data[0] - self.min_x) / self.res)
      col = int((data[1] - self.min_y) / self.res)
      self.grid[row, col] = data[2]

  def get_grid_diff(self):
    grid_diff_x1 = np.zeros(self.grid_size)
    grid_diff_x1[:self.grid_size[0] - 1, :] = self.grid[:self.grid_size[0] - 1, :]
    grid_diff_x2 = np.zeros(self.grid_size)
    grid_diff_x2[:self.grid_size[0] - 1, :] = self.grid[1:, :]
    grid_diff_x = abs(grid_diff_x2 - grid_diff_x1)

    grid_diff_y1 = np.zeros(self.grid_size)
    grid_diff_y1[:, :self.grid_size[1] - 1] = self.grid[:, :self.grid_size[1] - 1]
    grid_diff_y2 = np.zeros(self.grid_size)
    grid_diff_y2[:, :self.grid_size[1] - 1] = self.grid[:, 1:]
    grid_diff_y = abs(grid_diff_y2 - grid_diff_y1)
    self.grid_diff = grid_diff_x + grid_diff_y

  def get_local_minima(self):
    neighbors = np.array([[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]])
    num_neighbors = 8
    river_thres = 7
    low_height_thres = 15
    self.local_minima_list = []
    self.river_grid_list = []

    for r in range(self.grid_size[0]):
      for c in range(self.grid_size[1]):
        if (self.grid[r, c] < river_thres) & (self.grid[r, c] > 0):
          self.river_grid_list.append([r, c])
        if self.grid[r, c] == 0:
          continue
        is_local_minima = True
        for i in range(num_neighbors):
          neighbor = neighbors[i]
          if (r + neighbor[0] < 0) or (r + neighbor[0] >= self.grid_size[0]) or (c + neighbor[1] < 0) or (c + neighbor[1] >= self.grid_size[1]):
            continue
          if self.grid[r, c] > low_height_thres:
            is_local_minima = False
          if (self.grid[r, c] > self.grid[r + neighbor[0], c + neighbor[1]]) and (self.grid[r + neighbor[0], c + neighbor[1]] != -1):
            is_local_minima = False
        if is_local_minima:
          self.local_minima_list.append([r, c])
  
  def cluster_dfs(self, r, c):
    neighbors = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
    grid_diff_thres = 3
    height_thres = 15
    for dr, dc in neighbors:
      next_r = r + dr
      next_c = c + dc
      if (next_r < 0) or (next_r >= self.grid_size[0]) or (next_c < 0) or (next_c >= self.grid_size[1]):
        continue
      if (self.visited_grid[next_r, next_c] != -1) or (self.grid[next_r, next_c] == 0) or (self.grid[next_r, next_c] > height_thres):
        continue
      if abs(self.grid[next_r, next_c] - self.current_grid) > grid_diff_thres:
        continue
      self.visited_grid[next_r, next_c] = self.num_cluster
      self.cluster_dfs(next_r, next_c)


  def convert_grid_to_global_coord(self, coord, p1_type, p2_type):
    p1 = pyproj.Proj(init=p1_type)
    p2 = pyproj.Proj(init=p2_type)
    fx, fy = pyproj.transform(p1, p2, coord[0], coord[1])
    return np.dstack([fx, fy])[0][0]

  def get_cluster(self):
    self.visited_grid = np.full(self.grid_size, -1)
    self.num_cluster = 1
    for river_grid in self.river_grid_list:
      r, c = river_grid
      self.visited_grid[r, c] = -2
    
    for local_minima in self.local_minima_list:
      r, c = local_minima
      if (self.visited_grid[r, c] != -1):
        continue
      self.visited_grid[r, c] = self.num_cluster
      self.current_grid = self.grid[r, c]
      self.num_cluster_grids = 0
      self.cluster_dfs(r, c)

    self.marked_grid = np.zeros(self.grid_size)
    for r in range(self.grid_size[0]):
      for c in range(self.grid_size[1]):
        if self.visited_grid[r, c] > 0:
          self.marked_grid[r, c] = 1
  
  def get_polygons(self, use_csv = False, use_json = False):
    marked_img = np.array(self.marked_grid*255).astype('uint8')
    marked_img_cv = cv2.adaptiveThreshold(marked_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
    self.num_labels, self.labels = cv2.connectedComponents(marked_img_cv)
    polygon_idx = 0

    if use_json:
      contour_json = {}
      for i in range(self.num_labels):
        self.single_label = self.labels == i
        uint_img = np.array(self.single_label*255).astype('uint8')
        threshed = cv2.adaptiveThreshold(uint_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
        contour, _ = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contour[0]) <= 3:
          continue

        polygon_idx += 1
        contour_json[str(polygon_idx)] = []
        for cont in contour[0]:
          c = cont[0][0]
          r = cont[0][1]

          x = r * self.res + self.min_x
          y = c * self.res + self.min_y
          result = self.convert_grid_to_global_coord(np.array([x, y]), self.epsg, "epsg:4326")
          contour_json[str(polygon_idx)].append([result[0], result[1]])
      with open('result/polygons.json', 'w') as fp:
        json.dump(contour_json, fp, indent=4)

    contour_list = []
    if use_csv:
      for cont in contour[0]:
        c = cont[0][0]
        r = cont[0][1]
        contour_list.append([r, c])

      f = open("result/polygon%d.csv" %polygon_idx, 'w')
      wr = csv.writer(f)
      
      for points in contour_list:
        x = points[0] * self.res + self.min_x
        y = points[1] * self.res + self.min_y
        result = self.convert_grid_to_global_coord(np.array([x, y]), self.epsg, "epsg:4326")
        wr.writerow(result)
      f.close()
    
  def visualize(self):
    grid_color = self.grid.T / (self.max_z + 1e-9)
    grid_color = np.repeat(grid_color[:, :, np.newaxis], 3, axis=2)
    for local_minima in self.local_minima_list:
      grid_color[local_minima[1], local_minima[0], :] = np.array([1, 0, 0])

    for river in self.river_grid_list:
      grid_color[river[1], river[0], :] = np.array([0, 0, 1])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Height difference between nearest region')
    plt.imshow(self.grid_diff.T, origin='lower', vmax = 10)
    plt.colorbar(orientation='vertical')
    plt.savefig('figure/height_difference.png')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_title('Local minima and River')
    plt.imshow(grid_color, origin='lower', vmax = 1)
    plt.colorbar(orientation='vertical')
    plt.savefig('figure/local_minima.png')

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.set_title('Height')
    plt.imshow(self.grid.T, origin='lower', vmax = 30)
    plt.colorbar(orientation='vertical')
    plt.savefig('figure/height.png')

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.set_title('Clustered Regions')
    plt.imshow(self.visited_grid.T, origin='lower', vmax = 1)
    plt.colorbar(orientation='vertical')
    plt.savefig('figure/clustered_region.png')
    plt.show()

if __name__ == "__main__":
  sys.setrecursionlimit(5000)

  parser = argparse.ArgumentParser()
  parser.add_argument("-res", "--res", dest="res", type=int, default=90)          # extra value
  parser.add_argument("-vis", "--vis", dest="vis", type=str, default="false")
  parser.add_argument("-epsg", "--epsg", dest="epsg", type=int, default=5168)
  parser.add_argument("-save", "--save", dest="save", type=str)
  args = parser.parse_args()

  epsg = "epsg:" + str(args.epsg)
  res = args.res # meters
  visualize_result = False
  if args.vis == "true":
    visualize_result = True
  use_json, use_csv = False, False
  if args.save == "json":
    use_json = True
  elif args.save == "csv":
    use_csv = True

  detector = FloodRegionDetector(grid_resolution = res, epsg=epsg)
  detector.parse_data()
  detector.fill_grid()
  detector.get_grid_diff()
  detector.get_local_minima()
  detector.get_cluster()
  detector.get_polygons(use_csv=use_csv, use_json=use_json)
  if visualize_result:
    detector.visualize()
