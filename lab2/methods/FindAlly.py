from swiplserver import PrologThread

class FindAlly:
    def run(self, prolog: PrologThread):
        res = prolog.query(self.query())
        if not res or len(res) == 0:
            self.failure(res)
        else:
            self.success(res)
            
            
    def __init__(self, person: str):
        self.person = person
        
    def query(self):
        return f'alliance({self.person}, X)'
        
    def success(self, res):
        print(f'Найдено {len(res)} союзник(-ов) персонажа {self.person}:')
        for index, line in enumerate(res, 1):
            print(f'{index}.', line['X'])
        
    def failure(self, res):
        print(f'У {self.alliance} нет союзников.')