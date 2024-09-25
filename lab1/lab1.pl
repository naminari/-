%База знаний об игре Counter Strike Global Offensive


/*Факты*/

% Классы персонажей
character_class(warrior).
character_class(wizard).
character_class(rogue).
character_class(priest).
character_class(druid).
character_class(archer).
character_class(berserker).
character_class(paladin).

% Расы персонажей
character_race(human).
character_race(elf).
character_race(dwarf).
character_race(orс).
character_race(goblin).

% Факты о том, что персонаж принадлежит к классу (свойство персонажа)
character_class_of(arthur, warrior).
character_class_of(merlin, wizard).
character_class_of(lara, rogue).
character_class_of(ulfgar, priest).
character_class_of(faelar, druid).
character_class_of(elyan, archer).
character_class_of(goruk, berserker).
character_class_of(thorik, paladin).

% Факты о том, что персонаж принадлежит к расе (свойство персонажа)
character_race_of(arthur, human).
character_race_of(merlin, human).
character_race_of(lara, elf).
character_race_of(ulfgar, dwarf).
character_race_of(faelar, elf).
character_race_of(elyan, elf).
character_race_of(goruk, orc).
character_race_of(thorik, dwarf).

% Факты о поле персонажей (свойство персонажа)
character_sex(arthur, man).
character_sex(merlin, man).
character_sex(lara, woman).
character_sex(ulfgar, man).
character_sex(faelar, man).
character_sex(elyan, man).
character_sex(goruk, man).
character_sex(thorik, woman).

% Факты о наличии оружия у персонажей (отношение персонажа и оружия)
character_weapon(arthur, sword).
character_weapon(merlin, staff).
character_weapon(lara, dagger).
character_weapon(ulfgar, mace).
character_weapon(faelar, staff).
character_weapon(elyan, bow).
character_weapon(goruk, axe).
character_weapon(thorik, hammer).

% Факты о наличии заклинаний у персонажей (отношение персонажа и заклинания)
character_spell(merlin, fireball).
character_spell(merlin, shield).
character_spell(faelar, heal).
character_spell(ulfgar, smite).

% Факты о принадлежности персонажей к союзам (отношение персонажа и союза)
character_alliance(arthur, knights_of_valor).
character_alliance(merlin, circle_of_mages).
character_alliance(lara, shadow_blades).
character_alliance(ulfgar, stone_hammers).
character_alliance(faelar, emerald_enclave).
character_alliance(elyan, emerald_enclave).
character_alliance(goruk, bloodfang_clan).
character_alliance(thorik, stone_hammers).

% Факты о союзах персонажей (отношение двух персонажей)
alliance(arthur, merlin).
alliance(arthur, lara).
alliance(faelar, elyan).
alliance(ulfgar, thorik).

/* Правила */

% Правило о том, что персонаж является человеком и воином
is_human_warrior(Character) :-
    character_race_of(Character, human),
    character_class_of(Character, warrior).

% Правило о том, что персонаж является эльфом и магом или друидом
is_elf_magic_user(Character) :-
    character_race_of(Character, elf),
    (character_class_of(Character, wizard); character_class_of(Character, druid)).

% Правило о том, что персонаж обладает двуручным оружием (мечом, топором или молотом)
has_two_handed_weapon(Character) :-
    character_weapon(Character, sword);
    character_weapon(Character, axe);
    character_weapon(Character, hammer).

% Правило о том, что персонаж является воином и обладает двуручным оружием
is_warrior_with_two_handed_weapon(Character) :-
    character_class_of(Character, warrior),
    has_two_handed_weapon(Character).

% Правило о том, что персонаж может лечить (если он жрец или друид и знает заклинание "heal")
can_heal(Character) :-
    (character_class_of(Character, priest); character_class_of(Character, druid)),
    character_spell(Character, heal).

% Правило о том, что персонаж принадлежит к союзам рыцарей или магов
is_knight_or_mage(Character) :-
    character_alliance(Character, knights_of_valor);
    character_alliance(Character, circle_of_mages).

% Правило о том, что персонаж является дворфом и носит молот
is_dwarf_with_hammer(Character) :-
    character_race_of(Character, dwarf),
    character_weapon(Character, hammer).

% Правило о том, что персонаж может использовать заклинания щита (если персонаж маг и знает заклинание "shield")
can_cast_shield(Character) :-
    character_class_of(Character, wizard),
    character_spell(Character, shield).

% Правило о том, что персонаж принадлежит к союзу "Изумрудный Анклав" и может лечить
is_emerald_healer(Character) :-
    character_alliance(Character, emerald_enclave),
    can_heal(Character).

% Правило о том, что персонаж является орком и берсерком, но не состоит в союзе
is_orc_berserker_without_alliance(Character) :-
    character_race_of(Character, orc),
    character_class_of(Character, berserker),
    \+ character_alliance(Character, _).
