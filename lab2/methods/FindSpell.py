from swiplserver import PrologThread

class FindSpell:
    def run(self, prolog: PrologThread):
        res = prolog.query(self.query())
        if not res or len(res) == 0:
            self.failure(res)
        else:
            self.success(res)
            
            
    def __init__(self, wizard: str):
        self.wizard = wizard
        
    def query(self):
        return f'character_spell({self.wizard}, X)'
        
    def success(self, res):
        print(f'Найден {len(res)} спелл(-ов) персонажа {self.wizard}:')
        for index, line in enumerate(res, 1):
            print(f'{index}.', line['X'])
        
    def failure(self, res):
        print(f'У {self.wizard} нет магии.')