import numpy as np
import math
import matplotlib.pyplot as plt
from skimage import io
import glob2

ERROR_THRESHOLD = 0.1
def quiltSeaming(overlapSize, tile, mask, horizontalEdge=None, verticalEdge=None, row=False, col=False):
    if row:
        edgeSingleChannel = verticalEdge[:, :, 0]
        path = np.power(edgeSingleChannel[:, -overlapSize:] - tile[:, 0:overlapSize], 2).tolist()
        mask[:, 0:overlapSize] = minimumCostPath(path)

    if col:
        edgeSingleChannel = horizontalEdge[:, :, 0]
        path = np.power(edgeSingleChannel[-overlapSize:, :] - tile[0:overlapSize, :], 2).T.tolist()
        mask[0:overlapSize, :] = minimumCostPath(path).T

    return mask

def minimumCostPath(path):
    mask = np.ones(np.array(path).shape)
    rows , cols = len(path), len(path[0])

    for i in range(1, rows):
        path[i][0] +=  min(path[i - 1][0], path[i - 1][1])
        for j in range(1, cols - 1):
            path[i][j] += min(path[i - 1][j - 1], path[i - 1][j], path[i - 1][j + 1])
        path[i][cols - 1] +=  min(path[i - 1][cols - 2], path[i - 1][cols - 1])

    min_index = [0] * rows
    min_cost = min(path[-1])

    for index in range(1, cols - 1):
        if path[-1][index] == min_cost:
            min_index[-1] = index

    for i in range(rows - 2, -1, -1):
        j = min_index[i + 1]
        lower_bound = 0
        upper_bound = 1

        if j == cols - 1:
            lower_bound = cols - 2
            upper_bound = lower_bound + 1
        elif j > 0:
            lower_bound = j - 1
            upper_bound = lower_bound + 2

        min_cost = min(path[i][lower_bound:upper_bound + 1])

        for k in range(lower_bound, upper_bound + 1):
            if path[i][k] == min_cost:
                min_index[i] = k

    for i in range(rows):
        mask[i, :min_index[i]] = np.zeros(min_index[i])

    return mask

def matchBestTile(tiles, window):

    windowHeight, windowWidth, _ = window.shape
    errors = []
    mask = window != -1
    for i in range(tiles.shape[0]):
        tile = tiles[i, :, :, :]
        errors.append([np.sum((tile[:windowHeight, :windowWidth, :] - window) ** 2 * mask)])

    minSSD = np.min(np.array(errors))
    bestTiles = []
    for i, tile in enumerate(tiles):
        if errors[i] <= (1.0 + ERROR_THRESHOLD) * minSSD:
            bestTiles.append(tile[:windowHeight, :windowWidth, :])

    return bestTiles[np.random.randint(len(bestTiles))]

def imageQuilting(imgArray, tileSize, overlapSize, outSizeHeight, outSizeWidth):
    imageHeight, imageWidth, imageChannels = imgArray.shape
    tiles = []
    for i in range(imageHeight - tileSize):
        for j in range(imageWidth - tileSize):
            tiles.append(imgArray[i:i + tileSize, j:j + tileSize, :])

    tiles = np.array(tiles)
    outputImage = np.ones([outSizeHeight, outSizeWidth, imageChannels]) * -1
    outputImage[0:tileSize, 0:tileSize, :] = tiles[np.random.randint(len(tiles))]

    region = tileSize - overlapSize
    rows = 1 + int(np.ceil((outSizeHeight - tileSize) / region))
    cols = 1 + int(np.ceil((outSizeWidth - tileSize) / region))

    for row in range(rows):

        startX = row * region
        endX = min(startX + tileSize, outSizeHeight)

        for col in range(cols):
            if row == 0 and col == 0:
                continue

            startY = col * region
            endY = min(startY + tileSize, outSizeWidth)

            window = outputImage[startX:endX,startY:endY,:]

            bestTile = matchBestTile(tiles, window)
            tileSingleChannel = bestTile[:, :, 0]
            mask = np.ones(tileSingleChannel.shape)

            borderEndX = startX+overlapSize-1
            borderStartX = borderEndX-(bestTile.shape[0])+1

            borderEndY = startY + overlapSize - 1
            borderStartY = borderEndY - (bestTile.shape[1]) + 1

            h_edge = outputImage[borderStartX:borderEndX + 1, startY:endY, :]
            v_edge = outputImage[startX:endX, borderStartY:borderEndY + 1, :]

            if row == 0:
                mask = quiltSeaming(overlapSize, tileSingleChannel, mask, verticalEdge=v_edge, row=True)
            elif col == 0:
                mask = quiltSeaming(overlapSize, tileSingleChannel, mask, horizontalEdge=h_edge, col=True)
            else:
                mask = quiltSeaming(overlapSize, tileSingleChannel, mask, horizontalEdge=h_edge, verticalEdge=v_edge,
                                    row=True, col=True)

            mask = np.repeat(np.expand_dims(mask,axis=2),3,axis=2)
            maskNegate = mask==0

            outputImage[startX:endX,startY:endY,:] = maskNegate*window
            outputImage[startX:endX,startY:endY,:] = bestTile*mask+window

            completion = 100.0/rows*(row + col/cols)
            print("{0:.2f}% complete ...".format(completion), end="\r", flush=True)
            if endY == outSizeWidth:
                break

        if endX == outSizeHeight:
            print("100% complete!\n")
            break

    return outputImage/255.0

if __name__ == '__main__':
    imagePaths = glob2.glob("../data/texture/*")
    tileSize = 40
    for imagePath in imagePaths:
        img = io.imread(imagePath)
        img = img[:, :, :3]
        imageHeight, image_width, _ = img.shape
        newImageHeight, newImageWidth = int(3 * imageHeight), int(3 * image_width)
        overlap = math.ceil(tileSize / 6.0)

        quiltedImage = imageQuilting(img, tileSize, overlap, newImageHeight, newImageWidth)
        plt.imshow(quiltedImage)
        plt.title(f"tileSize {tileSize}, overlap {overlap}")
        plt.axis('off')
        filename = imagePath.split("/")[-1].split(".")[0]
        plt.savefig(f"../output/extra/{filename}_tileSize_{tileSize}_overlap_{overlap}.png")
        # plt.show()