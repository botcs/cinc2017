import requests

# sqlQuery: add or get data from database on the server 'users'
# e.g. sending the classification result of our neural network upon the class of a given recording:
# *UPDATE recordings SET machine_guess = "N" WHERE name = "A00002"*
# - Input: SQL statement
# - Output: list of dictionaries containing result of the query (dictionaries of list are the rows of result, keys of dictionaries are names of selected attributes of the query)


def sqlQuery(sql):
  r = requests.post("https://users.itk.ppke.hu/~hakta/challenge/query.php",
            data={'password': "qVK0fFt6zKLH{6T", 'sql': sql})
  if r.status_code != requests.codes.ok:
    raise Exception("ERROR " + str(r.status_code) + ": " + r.reason)
  else:
    try:
      resp = r.json()
      if resp["error"] != "OK":
        raise Exception(resp["error"][2])
      return resp["data"]
    except ValueError:
      print(r.text)
