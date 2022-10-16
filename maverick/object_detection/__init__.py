def get_line_thickness(image):
    return max(int(min((image.shape[1], image.shape[0])) / 150), 1)
