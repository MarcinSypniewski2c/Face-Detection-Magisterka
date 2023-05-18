from insightface.app import FaceAnalysis

class InsightFace:
    def __init__(self):
        self.model = FaceAnalysis(name="antelope")
        self.model.prepare(ctx_id=0, det_size=(320,320))

    def recognize(self, img):
        faces = self.model.get(img)
        return faces
    
    def get_embeddings(self, img):
        faces = self.recognize(img)
        embeds = []
        for face in faces:
            embeds.append(face[3])
        return embeds

