from .camvid_classes import CLASS_NAMES, CLASS_COLORS, NUM_CLASSES

print("Number of classes:", NUM_CLASSES)

print("\nClasses:")

for class_id, class_name in enumerate(CLASS_NAMES):
    print(class_id, "->", class_name)

print("\nNumber of RGB mappings:", len(CLASS_COLORS))