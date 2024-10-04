from swiplserver import PrologThread

class MainQuery:
    def run(self, prolog: PrologThread):
        res = prolog.query(self.query())
        if not res or len(res) == 0:
            self.failure(res)
        else:
            self.success(res)
            
            
    def __init__(self, name:str, class_character: str, weapon: str, allince: str):
        if class_character == "воин":
            class_character == 'warrior'
        else:
            class_character == 'wizard'
        if weapon == "двуручное_оружее":
            weapon == 'axe'
        else:
            weapon == 'staff'
        
        if allince == "мирный":
            allince = 'emerald_enclave'
        else:
            allince == 'knights_of_valor'
        self.name = name
        self.class_character = class_character
        self.weapon = weapon
        self.alliance = allince
        
    def query(self):
        return f'character_class_of(X, {self.class_character}), character_weapon(X, {self.weapon}), character_alliance(X, {self.alliance})'
        
    def success(self, res):
        print(f'{self.name[0]}, для вас подойдут данные герои:')
        for index, line in enumerate(res, 1):
            print(f'{index}.', line['X'])
        
    def failure(self, res):
        print(f'{self.name} для вас не найдено героев.')