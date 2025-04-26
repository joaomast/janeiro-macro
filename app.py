# Bibliotecas
from shiny.express import ui, render, input
from shiny.ui import page_navbar
from shiny import reactive
from shinyswatch import theme
import pandas as pd
import numpy as np
import plotnine as p9
import mizani as mi
from functools import partial
from faicons import icon_svg
import statsmodels.api as sm
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import VotingRegressor
from sklearn.svm import LinearSVR

# Funções
def preparar_dados(parquet, y):
    df_tmp = pd.read_parquet(parquet).reset_index(names = "data")
    ult = df_tmp.query("Tipo == @y").query("data == data.max()")
    df_tmp = pd.concat([
        df_tmp, 
        pd.DataFrame(
            {
                "data": ult.data.repeat(2), 
                "Valor": ult.Valor.repeat(2), 
                "Tipo": df_tmp.query("Tipo not in [@y, 'IA']").Tipo.unique().tolist(), 
                "Intervalo Inferior": ult.Valor.repeat(2), 
                "Intervalo Superior": ult.Valor.repeat(2)
            }
        )
        ])
    return df_tmp

def gerar_grafico(df, y, n, unidade, linha_zero = True):
    dt = input.periodo()
    mod = list(input.modelo())
    mod.insert(0, y)
    df_tmp = df.assign(
        Tipo = lambda x: pd.Categorical(x.Tipo, mod)
        ).query("data >= @dt and Tipo in @mod")

    def plotar_zero():
        if linha_zero:
            return p9.geom_hline(yintercept = 0, linetype = "dashed")
        else:
            return None
    
    def plotar_ic():
        if input.ic():
            return  p9.geom_ribbon(
                data = df_tmp,
                mapping = p9.aes(
                    ymin = "Intervalo Inferior",
                    ymax = "Intervalo Superior",
                    fill = "Tipo"
                    ),
                alpha = 0.25,
                color = "none",
                show_legend = False
            ) 
        else:
            return None
    
    plt = (
        p9.ggplot(df_tmp) +
        p9.aes(x = "data", y = "Valor", color = "Tipo") +
        plotar_zero() +
        plotar_ic() +
        p9.geom_line() +
        p9.scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
        p9.scale_y_continuous(breaks = mi.breaks.breaks_extended(6)) +
        p9.scale_color_manual(
            values = {
                y: "black", 
                "IA": "green",
                "Ridge": "blue",
                "Bayesian Ridge": "orange",
                "Huber": "red",
                "Ensemble": "brown"
                },
            drop = True,
            breaks = df_tmp.Tipo.unique().tolist()
            ) +
        p9.scale_fill_manual(
            values = {
                "IA": "green",
                "Ridge": "blue",
                "Bayesian Ridge": "orange",
                "Huber": "red",
                "Ensemble": "brown"
                },
            drop = True
            ) +
        p9.labs(color = "", x = "", y = unidade) +
        p9.theme(
            panel_grid_minor = p9.element_blank(),
            legend_position = "bottom"
            )
    )
    return plt

def gerar_cenarios_exogenas(df: pd.DataFrame, num_periodos: int, tipo_cenario: str, mm1y: int, freq: str) -> pd.DataFrame:

    if tipo_cenario not in ['zero', 'constante', 'media1y', 'linear']:
        raise ValueError(
            "O tipo de cenário deve ser 'Zero', 'Constante', "
            "'Média Móvel de 12 meses' ou 'Linear'."
        )

    ultimo_indice = df.index[-1]
    if freq == "MS":
        start = ultimo_indice + pd.offsets.MonthBegin(1) 
        periodos_futuros = pd.date_range(start=start, periods=num_periodos, freq=freq)
    else:
        start = ultimo_indice + pd.offsets.MonthBegin(3) 
        periodos_futuros = pd.date_range(start=start, periods=num_periodos/mm1y, freq=freq)
    
    df_futuro = pd.DataFrame(
        index=periodos_futuros, 
        columns=df.columns
        )

    df_completo = pd.concat([df, df_futuro])

    for col in df.columns:
        if tipo_cenario == 'zero':
            df_completo.loc[periodos_futuros, col] = 0
        elif tipo_cenario == 'constante':
            ultimo_valor = df.iloc[-1][col]
            df_completo.loc[periodos_futuros, col] = ultimo_valor
        elif tipo_cenario == 'media1y':
            if len(df) >= mm1y:
                media_movel = df[col].rolling(window=mm1y).mean().iloc[-1]
                df_completo.loc[periodos_futuros, col] = media_movel
            elif len(df) > 0:
                # Se tiver menos de 12 meses, usa a média disponível
                media_disponivel = df[col].mean()
                df_completo.loc[periodos_futuros, col] = media_disponivel
            else:
                df_completo.loc[periodos_futuros, col] = 0  # Se não houver dados, preenche com zero
        elif tipo_cenario == 'linear':
            if len(df) >= 2:
                valor_t_menos_1 = df.iloc[-2][col]
                valor_t = df.iloc[-1][col]
                incremento = valor_t - valor_t_menos_1
                valor_inicial = valor_t
                for i in range(num_periodos):
                    df_completo.loc[periodos_futuros[i], col] = valor_inicial + (i + 1) * incremento
            elif len(df) == 1:
                # Se houver apenas um ponto, assume incremento zero
                df_completo.loc[periodos_futuros, col] = df.iloc[-1][col]
            else:
                df_completo.loc[periodos_futuros, col] = 0  # Se não houver dados, preenche com zero

    return df_completo.loc[periodos_futuros]

# Função para transformar dados, conforme definido nos metadados
def transformar(x, tipo):

  switch = {
      "1": lambda x: x,
      "2": lambda x: x.diff(),
      "3": lambda x: x.diff().diff(),
      "4": lambda x: np.log(x),
      "5": lambda x: np.log(x).diff(),
      "6": lambda x: np.log(x).diff().diff()
  }

  if tipo not in switch:
      raise ValueError("Tipo inválido")

  return switch[tipo](x)

# Dados
df_ipca = preparar_dados("previsao/ipca.parquet", "IPCA")
df_cambio = preparar_dados("previsao/cambio.parquet", "Câmbio")
df_pib = preparar_dados("previsao/pib.parquet", "PIB")
df_selic = preparar_dados("previsao/selic.parquet", "Selic")

lista_modelos = list(set(
    df_ipca.query("Tipo != 'IPCA'").Tipo.unique().tolist() +
    df_cambio.query("Tipo != 'Câmbio'").Tipo.unique().tolist() +
    df_pib.query("Tipo != 'PIB'").Tipo.unique().tolist() +
    df_selic.query("Tipo != 'Selic'").Tipo.unique().tolist()
))

lista_variaveis = {
    "ipca": "Inflação (IPCA)",
    "cambio": "Taxa de Câmbio (BRL/USD)",
    "pib": "Atividade Econômica (PIB)",
    "selic": "Taxa de Juros (SELIC)"
}

# Layout
ui.page_opts(
    title = ui.span(
        ui.img(
            src = "https://aluno.analisemacro.com.br/download/59787/?tmstv=1712933415",
            height = 30,
            style = "padding-right:10px;"
            )
    ),
    window_title = "Painel de Previsões",
    page_fn = partial(page_navbar, fillable = True),
    theme = theme.minty
)

with ui.nav_panel("Painel de Previsões"):  
    with ui.layout_sidebar():
            
        # Inputs
        with ui.sidebar(width = 225):
            
            # Informação
            ui.markdown(
                (
                    "Acompanhe as previsões automatizadas dos principais " +
                    "indicadores macroeconômicos do Brasil e simule cenários" +
                    " alternativos em um mesmo dashboard."
                )
            )

            ui.input_selectize(
                id = "modelo",
                label = ui.strong("Selecionar modelos:"),
                choices = lista_modelos,
                selected = lista_modelos,
                multiple = True,
                width = "100%",
                options = {"plugins": ["clear_button"]}
                )
            ui.input_date(
                id = "periodo",
                label = ui.strong("Início do gráfico:"),
                value = pd.to_datetime("today") - pd.offsets.YearBegin(7),
                min = "2004-01-01",
                max = df_selic.data.max(),
                language = "pt-BR",
                width = "100%",
                format = "mm/yyyy",
                startview = "year"
                )
            ui.input_checkbox(
                id = "ic",
                label = ui.strong("Intervalo de confiança"),
                value = True
            )

            # Informação
            ui.markdown(
                "Elaboração: Fernando da Silva/Análise Macro"
            )
        
        # Outputs
        with ui.layout_column_wrap():
            
            with ui.navset_card_underline(title = ui.strong("Inflação (IPCA)")):
                with ui.nav_panel("", icon = icon_svg("chart-line")):
                    @render.plot
                    def ipca1():
                        plt = gerar_grafico(df_ipca, "IPCA", input.modelo(), "Var. %")
                        return plt

                with ui.nav_panel(" ", icon = icon_svg("table")):
                    @render.data_frame
                    def ipca2():
                        df_tmp = (
                            df_ipca
                            .query("Tipo != 'IPCA'")
                            .rename(
                                columns = {
                                    "Valor": "Previsão", 
                                    "Tipo": "Modelo", 
                                    "data": "Data"}
                                )
                            .assign(Data = lambda x: x.Data.dt.strftime("%m/%Y"))
                            .round(2)
                            .head(-2)
                        )
                        return render.DataGrid(df_tmp, summary = False)

            with ui.navset_card_underline(title = ui.strong("Taxa de Câmbio (BRL/USD)")):
                with ui.nav_panel("", icon = icon_svg("chart-line")):
                    @render.plot
                    def cambio1():
                        plt = gerar_grafico(df_cambio, "Câmbio", input.modelo(), "R\$/US\$", False)
                        return plt

                with ui.nav_panel(" ", icon = icon_svg("table")):
                    @render.data_frame
                    def cambio2():
                        df_tmp = (
                            df_cambio
                            .query("Tipo != 'Câmbio'")
                            .rename(
                                columns = {
                                    "Valor": "Previsão", 
                                    "Tipo": "Modelo", 
                                    "data": "Data"}
                                )
                            .assign(Data = lambda x: x.Data.dt.strftime("%m/%Y"))
                            .round(2)
                            .head(-2)
                        )
                        return render.DataGrid(df_tmp, summary = False)

        with ui.layout_column_wrap():
            
            with ui.navset_card_underline(title = ui.strong("Atividade Econômica (PIB)")):
                with ui.nav_panel("", icon = icon_svg("chart-line")):
                    @render.plot
                    def pib1():
                        plt = gerar_grafico(df_pib, "PIB", input.modelo(), "Var. % anual")
                        return plt

                with ui.nav_panel(" ", icon = icon_svg("table")):
                    @render.data_frame
                    def pib2():
                        df_tmp = (
                            df_pib
                            .query("Tipo != 'PIB'")
                            .rename(
                                columns = {
                                    "Valor": "Previsão", 
                                    "Tipo": "Modelo", 
                                    "data": "Data"}
                                )
                            .assign(Data = lambda x: x.Data.dt.to_period(freq = "Q").dt.strftime("T%q/%Y"))
                            .round(2)
                            .head(-2)
                        )
                        return render.DataGrid(df_tmp, summary = False)

            with ui.navset_card_underline(title = ui.strong("Taxa de Juros (SELIC)")):
                with ui.nav_panel("", icon = icon_svg("chart-line")):
                    @render.plot
                    def selic1():
                        plt = gerar_grafico(df_selic, "Selic", input.modelo, "% a.a.", False)
                        return plt

                with ui.nav_panel(" ", icon = icon_svg("table")):
                    @render.data_frame
                    def selic2():
                        df_tmp = (
                            df_selic
                            .query("Tipo != 'Selic'")
                            .rename(
                                columns = {
                                    "Valor": "Previsão", 
                                    "Tipo": "Modelo", 
                                    "data": "Data"}
                                )
                            .assign(Data = lambda x: x.Data.dt.strftime("%m/%Y"))
                            .round(2)
                            .head(-2)
                        )
                        return render.DataGrid(df_tmp, summary = False)


with ui.nav_panel("Simulador"):

    with ui.layout_sidebar():

        # Inputs
        with ui.sidebar(width = 250):
            
            ui.input_selectize(
                id = "variavel",
                label = ui.strong("Selecione a variável:"),
                choices = lista_variaveis,
                selected = "ipca",
                multiple = False,
                width = "100%"
                )
            
            ui.input_slider(
                id = "horizonte",
                label = ui.strong("Horizonte de previsão:"),
                min = 1,
                max = 12,
                value = 12,
                step = 1,
                width = "100%"
            )
            
            ui.input_radio_buttons(
                id = "cenario",
                label = ui.strong(" Tipo de cenário:"),
                choices = {
                    "zero": "Zero",
                    "constante": "Constante",
                    "media1y": "Média Móvel de 12 meses",
                    "linear": "Linear"
                },
                selected = "constante"
            )
  
            ui.input_date(
                id = "periodo_simulador",
                label = ui.strong("Início do gráfico:"),
                value = pd.to_datetime("today") - pd.offsets.YearBegin(7),
                min = "2004-01-01",
                max = df_selic.data.max() + pd.offsets.YearBegin(2),
                language = "pt-BR",
                width = "100%",
                format = "mm/yyyy",
                startview = "year"
                )

            ui.HTML(
                """
                <details>
                    <summary>Detalhes</summary>
                    <ul>
                        <li>Zero: cenário futuro das exógenas é preenchido com valor zero.</li>
                        <li>Constante: cenário futuro das exógenas é preenchido com o último valor disponível.</li>
                        <li>Média Móvel de 12 meses: cenário futuro das exógenas é preenchido com a média móvel dos últimos 12 meses disponíveis.</li>
                        <li>Linear: cenário futuro das exógenas é preenchido com o mesmo incremento das últimas duas observações disponíveis.</li>
                    </ul>
                <p>Obs.: unidade de medida pode conter transformações.
                </details>
                """
            )
        

        # Outputs
        with ui.card():

            ui.card_header(ui.span(icon_svg("chart-line"), " Gráfico simulado"))

            @render.plot
            def grafico_cenario():

                cores = {
                    "IPCA": "black", 
                    "Câmbio": "black", 
                    "PIB": "black", 
                    "Selic": "black", 
                    "Simulação": "blue",
                    }
                
                df = gerar_simulacao()

                dt = input.periodo_simulador()

                df = df.query("Data >= @dt")

                plt = (
                    p9.ggplot(df) +
                    p9.aes(x = "Data", y = "Valor", color = "Tipo") +
                    p9.geom_ribbon(
                        data = df,
                        mapping = p9.aes(
                            ymin = "Intervalo Inferior",
                            ymax = "Intervalo Superior",
                            fill = "Tipo"
                            ),
                        alpha = 0.25,
                        color = "none",
                        show_legend = False
                    ) +
                    p9.geom_line() +
                    p9.scale_x_date(date_breaks = "2 years", date_labels = "%Y") +
                    p9.scale_y_continuous(breaks = mi.breaks.breaks_extended(6)) +
                    p9.scale_color_manual(
                        values = cores,
                        drop = True,
                        breaks = df.Tipo.unique().tolist()
                        ) +
                    p9.scale_fill_manual(
                        values = cores,
                        drop = True
                        ) +
                    p9.labs(color = "", x = "", y = "") +
                    p9.theme(
                        panel_grid_minor = p9.element_blank(),
                        legend_position = "bottom"
                        )
                )
                
                return plt
                
        with ui.layout_column_wrap():

            @reactive.calc
            def prepara_xregs():

                # Planilha de metadados
                metadados = (
                    pd.read_excel(
                        io = "https://docs.google.com/spreadsheets/d/1x8Ugm7jVO7XeNoxiaFPTPm1mfVc3JUNvvVqVjCioYmE/export?format=xlsx",
                        sheet_name = "Metadados",
                        dtype = str,
                        index_col = "Identificador"
                        )
                    .filter(["Transformação"])
                )   

                xregs = {
                    "ipca": [
                        "expec_ipca_top5_curto_prazo", 
                        "ic_br", 
                        "cambio_brl_eur", 
                        "ipc_s"
                        ], #+ dummies_sazonais.columns.to_list() # + 1 lag
                    "cambio": [
                        "selic",
                        "expec_cambio",
                        "ic_br_agro",
                        "cotacao_petroleo_fmi"
                        ],# + 1 lag
                    "pib": [
                        "uci_ind_fgv",
                        "expec_pib",
                        "prod_ind_metalurgia"
                        ], #+ 2 lags
                    "selic": [
                        "selic",
                        "selic_lag2",
                        "pib_hiato",
                        "pib_hiato_lag1",
                        "inflacao_hiato"
                        ]
                }
                
                var = input.variavel()

                if var == "ipca":

                    inicio_treino = pd.to_datetime("2004-01-01")
                    
                    dados_brutos = pd.read_parquet("dados/df_mensal.parquet")

                    dados_tratados = dados_brutos.asfreq("MS")

                    y = dados_tratados.ipca.dropna()

                    x = dados_tratados.drop(labels = var, axis = "columns").copy()

                    x = x.filter(xregs[var])

                    for col in x.columns.to_list():
                        x[col] = transformar(x[col], metadados.loc[col, "Transformação"])

                    y = y[y.index >= inicio_treino]
                    
                    x = x.query("index >= @inicio_treino and index <= @y.index.max()")

                    x = x.bfill().ffill()

                elif var == "cambio":

                    inicio_treino = pd.to_datetime("2004-01-01")
                    
                    dados_brutos_m = pd.read_parquet("dados/df_mensal.parquet")

                    dados_tratados = (
                        dados_brutos_m
                        .asfreq("MS")
                        .rename_axis("data", axis = "index")
                    )

                    y = dados_tratados.cambio.dropna()

                    x = dados_tratados.filter(xregs[var]).copy()

                    for col in x.columns.to_list():
                        x[col] = transformar(x[col], metadados.loc[col, "Transformação"])

                    y = y[y.index >= inicio_treino]
                    
                    x = x.query("index >= @inicio_treino and index <= @y.index.max()")

                    x = x.bfill().ffill()

                elif var == "pib":
                    
                    inicio_treino = pd.to_datetime("1997-10-01")

                    dados_brutos_m = pd.read_parquet("dados/df_mensal.parquet")
                    dados_brutos_t = pd.read_parquet("dados/df_trimestral.parquet")

                    dados_tratados = (
                        dados_brutos_m
                        .resample("QS")
                        .mean()
                        .join(
                            other = (
                                dados_brutos_t
                                .set_index(pd.PeriodIndex(dados_brutos_t.index, freq = "Q").to_timestamp())
                                .resample("QS")
                                .mean()
                                ),
                            how = "outer"
                        )
                        .rename_axis("data", axis = "index")
                    )

                    dados_tratados.index = pd.to_datetime(dados_tratados.index)

                    y = dados_tratados.pib.dropna()

                    x = dados_tratados.filter(xregs[var]).copy()

                    for col in x.columns.to_list():
                        x[col] = transformar(x[col], metadados.loc[col, "Transformação"])

                    y = y[y.index >= inicio_treino]
                    
                    x = x.query("index >= @inicio_treino and index <= @y.index.max()")

                    x = x.bfill().ffill()
                else:

                    inicio_treino = pd.to_datetime("2004-01-01")

                    dados_brutos_m = pd.read_parquet("dados/df_mensal.parquet")
                    dados_brutos_a = pd.read_parquet("dados/df_anual.parquet")

                    dados_tratados = (
                        dados_brutos_m
                        .asfreq("MS")
                        .join(
                            other = dados_brutos_a.asfreq("MS").ffill(),
                            how = "outer"
                            )
                        .rename_axis("data", axis = "index")
                    )

                    y = dados_tratados.selic.dropna()

                    x = dados_tratados.drop(labels = "selic", axis = "columns").copy()

                    x_teorico = (
                        x
                        .copy()
                        .join(other = y, how = "outer")
                        .assign(
                            selic_lag1 = lambda x: x.selic.shift(1),
                            selic_lag2 = lambda x: x.selic.shift(2),
                            pib_potencial = lambda x: sm.tsa.filters.hpfilter(x.pib_acum12m.ffill(), 14400)[1],
                            pib_hiato = lambda x: (x.pib_acum12m / x.pib_potencial - 1) * 100,
                            pib_hiato_lag1 = lambda x: x.pib_hiato.shift(1),
                            inflacao_hiato = lambda x: x.expec_ipca_12m - x.meta_inflacao.shift(-12)
                        )
                        .filter(xregs[var])
                    )

                    y = y[y.index >= inicio_treino]
                    
                    x_teorico = x_teorico.query("index >= @inicio_treino and index <= @y.index.max()")

                    x_teorico = x_teorico.bfill().ffill()
                    
                    x = x_teorico.copy()
                
                return [x, y]

            @reactive.calc
            def cenario_tbl():

                var = input.variavel()

                if var == "pib":
                    cenario = gerar_cenarios_exogenas(
                        df = prepara_xregs()[0], 
                        num_periodos = input.horizonte(),
                        tipo_cenario = input.cenario(),
                        mm1y = 3,
                        freq = "QS-JAN"
                        )
                elif var == "ipca":
                    cenario = gerar_cenarios_exogenas(
                        df = prepara_xregs()[0], 
                        num_periodos = input.horizonte(),
                        tipo_cenario = input.cenario(),
                        mm1y = input.horizonte(),
                        freq = "MS"
                        )
                    dummies_sazonais = (
                        pd.get_dummies(cenario.index.month_name())
                        .astype(int)
                        .drop(labels = "December", axis = "columns")
                        .set_index(cenario.index)
                    )
                    cenario = cenario.join(other = dummies_sazonais, how = "outer")
                else:
                    cenario = gerar_cenarios_exogenas(
                        df = prepara_xregs()[0], 
                        num_periodos = input.horizonte(),
                        tipo_cenario = input.cenario(),
                        mm1y = input.horizonte(),
                        freq = "MS"
                        )
                
                return cenario

            @reactive.calc
            def gerar_simulacao():
                
                semente = 1984

                var = input.variavel()

                if var == "ipca":
                    modelo = ForecasterAutoreg(
                        regressor = Ridge(random_state = semente),
                        lags = 1,
                        transformer_y = PowerTransformer(),
                        transformer_exog = PowerTransformer()
                        )
                    modelo.fit(prepara_xregs()[1], prepara_xregs()[0])

                    previsao = (
                        modelo.predict_interval(
                            steps = input.horizonte(),
                            exog = cenario_tbl(),
                            n_boot = 500,
                            random_state = semente
                            )
                            .assign(Tipo = "Simulação")
                            .rename(
                            columns = {
                                "pred": "Valor", 
                                "lower_bound": "Intervalo Inferior", 
                                "upper_bound": "Intervalo Superior"
                                }
                            )
                        )
                    
                    df_fanchart = pd.concat([
                        df_ipca.query("Tipo == 'IPCA'"),
                        previsao.reset_index().rename(columns = {"index": "data"})
                        ]).rename(columns = {"data": "Data"})

                elif var == "cambio":
                    modelo = ForecasterAutoreg(
                        regressor = BayesianRidge(),
                        lags = 1,
                        transformer_y = PowerTransformer(),
                        transformer_exog = PowerTransformer()
                        )
                    modelo.fit(prepara_xregs()[1], prepara_xregs()[0])

                    previsao = (
                        modelo.predict_interval(
                            steps = input.horizonte(),
                            exog = cenario_tbl(),
                            n_boot = 500,
                            random_state = semente
                            )
                            .assign(Tipo = "Simulação")
                            .rename(
                            columns = {
                                "pred": "Valor", 
                                "lower_bound": "Intervalo Inferior", 
                                "upper_bound": "Intervalo Superior"
                                }
                            )
                        )
                    
                    df_fanchart = pd.concat([
                        df_cambio.query("Tipo == 'Câmbio'"),
                        previsao.reset_index().rename(columns = {"index": "data"})
                        ]).rename(columns = {"data": "Data"})
                    
                elif var == "pib":
                    modelo = ForecasterAutoreg(
                        regressor = Ridge(),
                        lags = 2,
                        transformer_y = PowerTransformer(),
                        transformer_exog = PowerTransformer()
                        )
                    modelo.fit(prepara_xregs()[1], prepara_xregs()[0])

                    previsao = (
                        modelo.predict_interval(
                            steps = int(input.horizonte()/3), # TODO: aperfeiçoar
                            exog = cenario_tbl(),
                            n_boot = 500,
                            random_state = semente
                            )
                            .assign(Tipo = "Simulação")
                            .rename(
                            columns = {
                                "pred": "Valor", 
                                "lower_bound": "Intervalo Inferior", 
                                "upper_bound": "Intervalo Superior"
                                }
                            )
                        )
                    
                    df_fanchart = pd.concat([
                        df_pib.query("Tipo == 'PIB'"),
                        previsao.reset_index().rename(columns = {"index": "data"})
                        ]).rename(columns = {"data": "Data"})
                
                else:
                    
                    modelo = ForecasterAutoreg(
                        regressor = VotingRegressor([
                            ("bayes", BayesianRidge()),
                            ("svr", LinearSVR(random_state = semente, dual = True, max_iter = 100000)),
                            ("ridge", Ridge(random_state = semente))
                            ]),
                        lags = 2,
                        transformer_y = PowerTransformer(),
                        transformer_exog = PowerTransformer()
                        )
                    modelo.fit(prepara_xregs()[1], prepara_xregs()[0])

                    previsao = (
                        modelo.predict_interval(
                            steps = input.horizonte(),
                            exog = cenario_tbl(),
                            n_boot = 500,
                            random_state = semente
                            )
                            .assign(Tipo = "Simulação")
                            .rename(
                            columns = {
                                "pred": "Valor", 
                                "lower_bound": "Intervalo Inferior", 
                                "upper_bound": "Intervalo Superior"
                                }
                            )
                        )
                    
                    df_fanchart = pd.concat([
                        df_selic.query("Tipo == 'Selic'"),
                        previsao.reset_index().rename(columns = {"index": "data"})
                        ]).rename(columns = {"data": "Data"})
                
                return df_fanchart


            with ui.layout_columns():

                with ui.card():
                    ui.card_header(ui.span(icon_svg("table"), " Cenário simulado"))
                    @render.data_frame
                    def tabela_cenarios():
                        return render.DataGrid(
                            (
                                cenario_tbl()
                                .reset_index()
                                .rename(columns = {"index": "Data"})
                                .assign(Data = lambda x: x.Data.dt .strftime("%Y-%m"))
                                .round(2)
                            ),
                            summary = False
                            )

                with ui.card():
                    ui.card_header(ui.span(icon_svg("table"), " Valores simulados"))
                    @render.table
                    def tabela_cenario():
                        df = (
                            gerar_simulacao()
                            .query("Tipo == 'Simulação'")
                            .assign(
                                Data = lambda x: x.Data.dt.strftime("%Y-%m")
                                )
                            .round(2)
                        )
                        return df
