
from helper.text_decorators import convert_english_to_persian, split_string_language_specific




class Entries:

    def __init__(self, platePercent,charPercent,eDate,eTime,plateNum,status,imgpath,scrnpath,isarvand,rtpath,):
        self.status = status
        self.plateNum = plateNum
        self.eTime = eTime
        self.eDate = eDate
        self.charPercent = charPercent
        self.platePercent = platePercent
        self.impath=imgpath
        self.scrpath=scrnpath,
        self.isarvand=isarvand,
        self.rtpath=rtpath

    def getTime(self):
        return self.eTime

    def getDate(self, persian=True):
       
        return self.eDate

    def getPlatePic(self):

        
        
        return f'{self.impath}'

    def getCharPercent(self):
        return "{}%".format(self.charPercent)

    def getPlatePercent(self):
        return "{}%".format(self.platePercent)

    def getPlateNumber(self, display=False):
        return convert_english_to_persian(split_string_language_specific(self.plateNum), display)




