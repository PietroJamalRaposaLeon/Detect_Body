import cv2


# Crie nosso classificador de corpos
vid = cv2.VideoCapture(0)

# Inicie a captura de vídeo para o arquivo de vídeo
cap = cv2.VideoCapture('walking.avi')

# Faça o loop assim que o vídeo for carregado com sucesso
while True:
    
    # Leia o primeiro quadro
    ret, frame = cap.read()

    # Converta cada quadro em escala de cinza
    ret, frame = vid.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = vid.detectMultiScale(gray)

    for (x, y,  l, a,) in faces:
        cv2.rectangle(frame, (x, y), (x+l, y+a), (0,0,255), 2)


    # Exiba o quadro resultante
    cv2.imshow("Web cam", frame)
    # Passe o quadro para nosso classificador de corpos
    body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')
    
    # Extraia as caixas delimitadoras para quaisquer corpos identificados
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)

    if cv2.waitKey(1) == 32: #32 é a barra de espaço
        break

cap.release()
cv2.destroyAllWindows()
