from swiplserver import PrologThread

class GetClanMembers:
    def run(self, prolog: PrologThread):
        res = prolog.query(self.query())
        if not res or len(res) == 0:
            self.failure(res)
        else:
            self.success(res)
            
            
    def __init__(self, alliance: str):
        self.alliance = alliance
        
    def query(self):
        return f'character_alliance(X, {self.alliance})'
        
    def success(self, res):
        print(f'Найдено {len(res)} участника(-ов) альянса {self.alliance}:')
        for index, line in enumerate(res, 1):
            print(f'{index}.', line['X'])
        
    def failure(self, res):
        print(f'В {self.alliance} никто не вступил.')