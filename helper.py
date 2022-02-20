def addToAverages(avg_array, obj):
    avg_array[0] += obj['emotion']['angry']
    avg_array[1] += obj['emotion']['disgust']
    avg_array[2] += obj['emotion']['fear']
    avg_array[3] += obj['emotion']['happy']
    avg_array[4] += obj['emotion']['sad']
    avg_array[5] += obj['emotion']['surprise']
    avg_array[6] += obj['emotion']['neutral']
