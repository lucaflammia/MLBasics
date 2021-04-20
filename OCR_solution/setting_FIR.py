import sqlite3

BASEPATH = r'C:\Users\Utente\Documents\Multitraccia\Progetti\Cobat\OCR_development'
IMAGEPATH = r'C:\Users\Utente\Documents\Multitraccia\Progetti\Cobat\OCR_development\FIR_test\FIR_A3P'

INFO_FIR = {
    'PRODUTTORE': {
        'NOWORD': ['produttore', 'ragione', 'sociale', 'denominazione', 'unita']
    },
    'TRASPORTATORE': {
                'NOWORD': ['trasportatore', 'ragione', 'sociale', 'denominazione', 'luogo', 'indirizzo']
            },
    'RACCOGLITORE': {
            'NOWORD': ['ragione', 'sociale', 'denominazione', 'luogo', 'destinazione']
        }
}

# TIPO_A = {
#     'TEXT': ["formulario", "rifiuti"],
#     'NOWORD': ["identificazione"],
#     'SIGN': ["<", "<"],
#     'FILES': []
# }

# TIPO_B = {
#     'TEXT': ["recycling", "systems"],
#     'SIGN': ["<", "<"],
#     'FILES': []
# }

# TIPO_C = {
#     'TEXT': ["trasporto", "rifiuti"],
#     'SIGN': [">", "<"],
#     'FILES': []
# }

TIPO_FIR = {
    'TIPO_A': {
        'TEXT': ["formulario", "rifiuti"],
        'NO_WORD': ["identificazione"],
        'SIGN': ["<", "<"],
        'FILES': []
    },
    
    # 'TIPO_B': TIPO_B,
    # 'TIPO_C': TIPO_C,
    'NC': []
}


class Database:
    def __init__(self, db):
        self.conn = sqlite3.connect(db)
        self.cur = self.conn.cursor()
        self.tb_files = """
        CREATE TABLE if not exists files_A3P 
        (id INTEGER PRIMARY KEY AUTOINCREMENT, file VARCHAR(50) NOT NULL, 
        tipologia VARCHAR(50) NOT NULL, ts TIMESTAMP);
        """
        self.tb_parole = """CREATE TABLE if not exists parole_A3P (  
        id INTEGER PRIMARY KEY AUTOINCREMENT, 
        parola VARCHAR(255) NOT NULL, 
        coor_x FLOAT(10,5) NOT NULL, 
        coor_y FLOAT(10,5) NOT NULL, 
        id_file INTEGER NOT NULL, 
        div VARCHAR(255) NOT NULL, 
        dpi INTEGER NOT NULL, 
        flt VARCHAR(255) NOT NULL, 
        ts TIMESTAMP, 
        FOREIGN KEY (id_file) REFERENCES files_A3P (id) );"""
        self.cur.execute(self.tb_files)
        self.cur.execute(self.tb_parole)
        self.conn.commit()

    def __del__(self):
        self.conn.close()

class QueryFir:
    def __init__(self):
        self.body = """
                    SELECT parola, coor_x, coor_y, file
                    FROM parole_A3P p
                    LEFT JOIN "files_A3P" f
                    ON (p.id_file=f.id)
                """
        self.sub_body = """
                    SELECT p.id
                    FROM parole_A3P p
                    LEFT JOIN files_A3P f
                    ON (p.id_file=f.id)
                """