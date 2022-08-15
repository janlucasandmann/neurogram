# Marketing Tags Revamp
- tool-agnostische Lösung, die in verschiedenen Tag Managern angewandt werden kann, aber trotzdem zum Großteil zentral gepflegt wird (toolspezifische Umsetzungen außen vor)
- zentrale Konfigurationsdatei, die auf AWS gehostet wird, und über die mit zentralen Funktionen Regeln in Launch/GTM gebaut werden können
- Prüfung des Consents / andere konkrete Bedingungen für die Marketing Tags (z.B. für welches Event, …) sind nicht Teil der Konfigurationsdatei, sondern zwingend in den Rules zu berücksichtigen → hierfür können zentrale Funktionen bspw. über Launch Extension zur Verfügung gestellt werden (Consent Extension + Marketing Tag Extension, die nur bei Consent lädt)
- Umgang mit der zentralen Konfigurationsdatei:
    - Rule, die als allererstes feuert, um zu checken, ob es für den Markt (= Property) überhaupt Marketing Tags gibt, die eingebaut werden müssen
    - Für jeden einzubauenden Marketing Tag werden dann zentrale Funktionen für diesen Tag geladen (lässt sich sehr wahrscheinlich nicht zusammenfassen, weil Unterschiede zwischen den Anbietern)
- Konfigurationsdatei sollte je Markt, je Product und je Anbieter über die API abfragbar sein (eigene Methoden oder Parameter), um nicht unnötigen Kram zu laden, der den Markt oder das Product gar nicht betreffen
- Zentrale Funktionen um Marketing Tags zu implementieren (auf Konfigurationsdatei basierend)
```jsx
const config = {
  NBW: {
    disabled: true, // flag can be set to disable all marketing tags instead of deleting it completely
    facebook: {
      disabled: true, // flag can be set to disable the tag instead of deleting it completely
      id: "444962119954478", // id used in fbq('init', id)
      dontTrackPageView: true, // flag can be set when fbq('track', 'PageView') should NOT be fired with init
      trigger: "pageView",
      conditions: [
        {
          type: "href", // "domain", "pathname", "pageName", "brand", "eventType", "eventAction"
          compareMethod: "contains", // "exact match", "starts with", "regex", "is true", "is false"
          value: "solution-noleggio.vwfs.it", // can also be an array of strings
          operator: "or", // "and"; only needed, if value is an array
        },
        {
          type: "pathname",
          compareMethod: "exact match",
          value: "/it",
        },
      ],
      conditionsOperator: "and", // "or"; only needed, if more than one object in conditions
      events: [
        {
          value: "CompleteRegistration",
          trigger: "pageView", // "viewChange", "interaction"
          conditions: [
            {
              type: "href", // "domain", "pathname", "pageName", "eventType", "eventAction"
              compareMethod: "exact match",
              value: "https://vwfs.de/kategorie/seite",
            },
          ],
          data: {
            value: 1,
            currency: "EUR",
            content_name: "vw_fs",
            content_category: "solution",
          },
          disabled: true, // flag can be set to disable the tag instead of deleting it completely
        },
      ],
    },
  },
  "Service and inspection": {
    adform: {
      trigger: "pageView",
      conditionsOperator: "and",
      events: [
        {
          conditions: [
            {
              type: "brand",
              compareMethod: "exact match",
              value: "audi",
            },
            {
              type: "pageName",
              compareMethod: "exact match",
              value: "home",
            },
          ],
          conditionsOperator: "and",
          data: {
            HttpHost: "server.adform.net",
            pm: 1882177,
            divider: "|",
            pagename: "Audi|WuI|LP(LP_Audi)",
            order: {
              sv6: null,
              sv7: null,
              sv8: "Service and Inspection",
            },
          },
        },
        {
          conditions: [
            {
              type: "brand",
              compareMethod: "exact match",
              value: "audi",
            },
            {
              type: "pageName",
              compareMethod: "exact match",
              value: "Product Page",
            },
          ],
          conditionsOperator: "and",
          data: {
            HttpHost: "server.adform.net",
            pm: 1882177,
            divider: "|",
            pagename: "Audi|WuI|Angebot (Angebot_Audi)",
            order: {
              sv6: null,
              sv7: null,
              sv8: "Service and Inspection",
            },
          },
        },
      ],
    },
  },
  "Motor insurance": {
    facebook: {
      disabled: true, // flag can be set to disable the tag instead of deleting it completely
      id: "444962119954478", // id used in fbq('init', id)
      dontTrackPageView: true, // flag can be set when fbq('track', 'PageView') should NOT be fired with init
      trigger: "pageView",
      conditions: [
        {
          type: "href", // "domain", "pathname", "pageName", "brand", "eventType", "eventAction"
          compareMethod: "contains", // "exact match", "starts with", "regex", "is true", "is false"
          value: "solution-noleggio.vwfs.it", // can also be an array of strings
          operator: "or", // "and"; only needed, if value is an array
        },
        {
          type: "pathname",
          compareMethod: "exact match",
          value: "/it",
        },
      ],
      conditionsOperator: "and", // "or"; only needed, if more than one object in conditions
      events: [
        {
          value: "CompleteRegistration",
          trigger: "pageView", // "viewChange", "interaction"
          conditions: [
            {
              type: "href", // "domain", "pathname", "pageName", "eventType", "eventAction"
              compareMethod: "exact match",
              value: "https://vwfs.de/kategorie/seite",
            },
          ],
          data: {
            value: 1,
            currency: "EUR",
            content_name: "vw_fs",
            content_category: "solution",
          },
          disabled: true, // flag can be set to disable the tag instead of deleting it completely
        },
      ],
    },
  },
  CUSTOMERPORTAL: {},
};
const hasFacebookTags = (config.NBW.facebook && !config.NBW.facebook.disabled) || false;
```
1. Rule #1 (”Check Marketing Tag config”) – 1x
    1. feuert bei jedem pageView / viewChange (früh)
    2. Konfigurationsobjekt des Markets über API laden
    3. Prüfen, für welche Marketing Tags es Einträge gibt
        1. Information auch in DEs o.Ä. schreiben, damit bei nachträglichem Vergeben von Consent trotzdem Rule #2 getriggert werden kann → entweder über Data Element Changed (suboptimal) oder Prüfung beim Consent, ob für die Consent-Kategorie auch Tags vorhanden sind und Direct Calls feuern
    4. Direct Call je Tag für Rule #2 feuern, ggfs. gewrappt in Condition (wenn nicht für gesamtes Produkt gelten soll), Konfigurationsobjekt des Tags als Payload
2. Rule #2 (”Load Marketing Tag functions and load global tag”) – 1x pro Tag
    1. je Marketing Tag Anbieter eine Rule, die zentrale Funktionen für den Einbau der Marketing Tags lädt
    2. nur feuern, wenn auch Consent für den Marketing Tag gegeben
    3. Zentrale Funktionen, die bereit gestellt werden müssen
        1. Umgang mit Konfigurationsobjekt (z.B. Prüfung auf bestimmte Properties, Prüfung der Conditions, …)
        2. Funktionen zum Aufruf der tagspezifischen Tracking-Funktionalitäten
            1. berücksichtigen, dass z.B. gtag o.Ä. nur einmal geladen werden (→ SPAs)
    4. Global Tag laden (fbq(init), gtag, …)
    5. Für jedes zu trackende Event einen Direct Call feuern, der in die jeweiligen Conditions gewrappt ist (damit nicht ständig rules feuern, die geprüft werden müssen)
3. Rule #3 (”Track Marketing Tag event”) – 1x pro Tag
    1. evtl. Consent- und Event-Conditions (eigentlich nicht nötig, weil nur getriggert wenn Rule #2 feuert (= Consent da) und Condition für das Event greift (= Direct Call))
    2. eigentlicher Einbau der Marketing Tags auf Basis der bereitgestellten Funktionen
    3. Möglichkeit, die u.U. mitgelieferte ``data``-Property zu überschreiben und/oder zu erweitern (Object spread)
4. Weitere Rules, die nicht direkt feuern können, weil sie auf einem anderen Event basieren (z.B. Interaktion mit Button etc.), müssen manuell angelegt werden
→ Rule #3 + weitere Rules könn(t)en entfallen, wenn man verlässliche Events für Macro Conversions hat
