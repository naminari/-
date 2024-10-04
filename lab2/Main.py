from swiplserver import PrologMQI, create_posix_path
from methods import FindAlly, FindSpell, GetClanMembers, MainQuery

import re

KNOWLEDGE_BASE_PATH = r'C:\Users\User\Downloads\лабс\сииии\лапка2\lab1.pl'

queries = [
    "\n * Выведи имена участников клана number_k (knights_of_valor, circle_of_mages, shadow_blades, stone_hammers, emerald_enclave, emerald_enclave,  bloodfang_clan, stone_hammers)",
    "Какое заклинание знает number_k? (merlin, faelar, ulfgar)",
    "Есть ли у number_k союзник (arthur, merlin, lara, ulfgar, faelar, elyan, goruk, thorik)",
    "Мой персонаж leo, я - воин/маг, люблю двуручное_оружее/магию  и хотел бы мирный/воинственный клан"
]

patterns = {
    r'Выведи имена участников клана (.+)': GetClanMembers.GetClanMembers,
    r'Какое заклинание знает (.+)\?': FindSpell.FindSpell,
    r'Есть ли у (.+) союзник\?': FindAlly.FindAlly,
    "Мой персонаж (.+), я - (.+), люблю (.+)  и хотел бы (.+) клан": MainQuery.MainQuery

}

with PrologMQI() as mqi:
    with mqi.create_thread() as prolog:
        path = create_posix_path(KNOWLEDGE_BASE_PATH)
        prolog.query(f'consult("{path}")')
        print("Успешно загружена база знаний Prolog!")
        print("\nПримеры запросов, которые вам доступны:", "\n * ".join(queries))
        print("\nДля завершения введите - exit.")
        while True:
            query = input('$ ')
            if query.lower() == 'exit':
                break

            for pattern in patterns:
                match = re.match(pattern, query, re.IGNORECASE)
                if match is None:
                    continue
                processor = patterns[pattern](*match.groups())
                processor.run(prolog)
                break
            else:
                print("Неверный запрос")