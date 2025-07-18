import streamlit as st
import pandas as pd
import polars as pl
import plotly.express as px
import io
import numpy as np

st.set_page_config(layout="wide")
st.title('Dashboard Trades de Robots (EAs) - Drawdown máximo real por EA')
st.markdown("""
Sube uno o varios archivos Excel o CSV de trades (puedes exportarlos de MetaTrader, cTrader, etc).

- Todas las métricas y KPIs usan **Profit total neto = Profit - |Comisiones+Tasas+Swap|** multiplicado si lo deseas.
""")

def cargar_y_limpiar_datos(uploaded_files):
    dfs = []
    columnas_necesarias = [
        "Ticket", "Open Time", "Type", "Size", "Item", "Price", "S / L", "T / P",
        "Close Time", "Close Price", "Commission", "Taxes", "Swap", "Profit",
        "MagicNumber", "Coment", "Cleaned_EA"
    ]
    tipos_forzados = {
        "Ticket": pl.Int64,
        "Size": pl.Float64,
        "Price": pl.Float64,
        "S / L": pl.Float64,
        "T / P": pl.Float64,
        "Close Price": pl.Float64,
        "Commission": pl.Float64,
        "Taxes": pl.Float64,
        "Swap": pl.Float64,
        "Profit": pl.Float64,
        "MagicNumber": pl.Int64,
    }
    for uf in uploaded_files:
        try:
            is_csv = uf.name.lower().endswith('.csv')
            if is_csv:
                uf.seek(0)
                content = uf.read()
                uf.seek(0)
                try:
                    content_str = content.decode("utf-8")
                except UnicodeDecodeError:
                    content_str = content.decode("latin1")
                head = content_str.split('\n',1)[0]
                delimiter = ';' if head.count(';') > head.count(',') else ','
                df_part = pd.read_csv(io.StringIO(content_str), delimiter=delimiter)
            else:
                df_part = pd.read_excel(uf, sheet_name=1, engine='openpyxl')

            if "Price.1" in df_part.columns:
                df_part = df_part.rename(columns={"Price.1": "Close Price"})
            price_locs = [i for i, c in enumerate(df_part.columns) if c == "Price"]
            if len(price_locs) > 1:
                cols = list(df_part.columns)
                cols[price_locs[1]] = "Close Price"
                df_part.columns = cols

            if "Cleaned_EA" not in df_part.columns:
                posibles = [c for c in df_part.columns if c.lower() in ['coment','comment','ea','magic','magicnumber']]
                si_ea = 'Coment' if 'Coment' in df_part.columns else (posibles[0] if posibles else None)
                if si_ea:
                    df_part["Cleaned_EA"] = df_part[si_ea].astype(str)
                else:
                    df_part["Cleaned_EA"] = "EA_desconocido"

            df_part = pl.from_pandas(df_part)
            for col, tpo in tipos_forzados.items():
                if col in df_part.columns:
                    df_part = df_part.with_columns(
                        [pl.col(col).cast(tpo, strict=False).alias(col)]
                    )
            falta_col = [col for col in columnas_necesarias if col not in df_part.columns]
            if falta_col:
                st.error(f"El archivo {uf.name} no tiene todas las columnas requeridas. "
                         f"Las que faltan: {falta_col}")
                return None
            dfs.append(df_part)
        except Exception as e:
            st.error(f"No se pudo leer el archivo {uf.name}. Error: {e}")
            return None
    if dfs:
        df = pl.concat(dfs)
        return df
    else:
        st.error("Ningún archivo válido se pudo cargar.")
        return None

def calcular_drawdown_maximo_curva(series_profit):
    valores = np.array(series_profit)
    if valores.size == 0:
        return np.nan
    saldo = valores.cumsum()
    peak = np.maximum.accumulate(saldo)
    dd = saldo - peak
    return dd.min() if dd.size > 0 else np.nan

def calcular_resumen(df, multiplicador=1.0):
    df = df.with_columns([
        (pl.col('Profit') * multiplicador).alias('Profit_neto'),
        pl.col('Close Time').dt.strftime('%Y-%m').alias('Mes')
    ])
    meses_disponibles = df.select('Mes').unique().sort('Mes')['Mes'].to_list()
    resumen_lista = []
    eas_unicos = df.select('Cleaned_EA').unique()['Cleaned_EA'].to_list()
    for ea in eas_unicos:
        dfg = df.filter(pl.col('Cleaned_EA') == ea)
        if dfg.height == 0:
            continue
        profit_total = dfg['Profit_neto'].sum()
        dd_max = calcular_drawdown_maximo_curva(dfg['Profit_neto'].to_numpy())
        trades = dfg.height
        if 'Mes' in dfg.columns and dfg.height > 0:
            grupos_df = dfg.group_by('Mes').agg(pl.col('Profit_neto').sum().alias('Profit_mes'))
            grupos = grupos_df.to_dict(as_series=False)
            profit_por_mes = dict(zip(grupos['Mes'], grupos['Profit_mes']))
        else:
            profit_por_mes = {}
        profit_positivo = dfg.filter(pl.col('Profit_neto') > 0)['Profit_neto'].sum()
        profit_negativo = dfg.filter(pl.col('Profit_neto') < 0)['Profit_neto'].sum()
        profit_factor = profit_positivo / abs(profit_negativo) if profit_negativo != 0 else np.nan
        win_ratio = dfg.filter(pl.col('Profit_neto') > 0).height / trades if trades > 0 else 0
        avg_trade = dfg['Profit_neto'].mean() if trades > 0 else 0
        ret_dd = profit_total / abs(dd_max) if (dd_max not in (0, np.nan, None) and not np.isnan(dd_max)) else np.nan
        fila = {
            'Cleaned_EA': ea,
            'Profit_Total': profit_total,
            'Ret/DD': ret_dd,
            'DD_Max_Curva': dd_max,
            'Trades': trades,
            'Profit_Factor': profit_factor,
            'Win_Ratio_%': 100 * win_ratio,
            'Avg_Profit_Trade': avg_trade
        }
        for mes in meses_disponibles:
            fila[f'Profit_{mes}'] = profit_por_mes.get(mes, 0)
        resumen_lista.append(fila)
    resumen = pl.DataFrame(resumen_lista)
    return resumen, meses_disponibles

def mostrar_kpis(resumen, dd_max_portafolio, df_portafolio, multiplicador):
    total_profit = (df_portafolio['Profit'] * multiplicador).sum()
    total_commissions = (
        (df_portafolio['Commission'] * multiplicador).sum() +
        (df_portafolio['Taxes'] * multiplicador).sum() +
        (df_portafolio['Swap'] * multiplicador).sum()
    )
    profit_total_neto = total_profit - abs(total_commissions)
    total_trades = len(df_portafolio)
    win_ratio = resumen['Win_Ratio_%'].to_numpy()
    trades_ea = resumen['Trades'].to_numpy()
    win_ratio_avg = np.average(win_ratio, weights=trades_ea) if total_trades > 0 else 0
    ret_dd_global = profit_total_neto / abs(dd_max_portafolio) if dd_max_portafolio != 0 else np.nan

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Profit total neto", f"{profit_total_neto:,.2f}")
    kpi2.metric("Total Trades", int(total_trades))
    kpi3.metric("Win Ratio Promedio (%)", f"{win_ratio_avg:,.1f}")
    kpi4.metric("Ret/DD Global", f"{ret_dd_global:,.2f}" if not np.isnan(ret_dd_global) else "N/A")

def descargar_dataframe(df, nombre_archivo, label):
    towrite = io.BytesIO()
    df.to_pandas().to_excel(towrite, index=False, engine='openpyxl')
    towrite.seek(0)
    st.download_button(
        label=label,
        data=towrite,
        file_name=nombre_archivo,
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

uploaded_files = st.file_uploader(
    "Carga uno o varios archivos Excel o CSV de trades",
    type=['xlsx','xls','csv'],
    accept_multiple_files=True,
    help="Puedes seleccionar varios archivos para combinar los datos."
)

if uploaded_files:
    df = cargar_y_limpiar_datos(uploaded_files)
    if df is None:
        st.stop()
    if 'Cleaned_EA' not in df.columns:
        st.error("La columna Cleaned_EA no existe en los datos.")
        st.stop()
    with pl.StringCache():
        df = df.with_columns([
            pl.col('Open Time').str.strptime(pl.Datetime, format="%Y.%m.%d %H:%M:%S", strict=False).alias('Open Time'),
            pl.col('Close Time').str.strptime(pl.Datetime, format="%Y.%m.%d %H:%M:%S", strict=False).alias('Close Time'),
            pl.col('Cleaned_EA').cast(pl.Utf8).alias('Cleaned_EA'),
        ])
        df = df.with_columns([
            pl.col('Close Time').dt.strftime('%Y-%m').alias('Mes')
        ])
        excluir_comentarios = ["deposited", "deposit", "withdraw", "withdrawal", "api", "canceled", "cancelado"]
        df = df.filter(
            pl.col('Coment').is_not_null() &
            ~pl.col('Coment').str.to_lowercase().str.contains('|'.join(excluir_comentarios))
        )

    num_vacios = df.select((pl.col('Cleaned_EA').str.strip_chars() == "").sum()).item()
    if num_vacios > 0:
        st.warning(f"⚠️ Hay {num_vacios} filas con Cleaned_EA vacío. Se recomienda limpiar los encabezados o los datos.")

    eas_limpios = df.select(pl.col('Cleaned_EA').str.strip_chars().str.to_lowercase().alias('ea_limpio'))['ea_limpio']
    n_unique = eas_limpios.n_unique()
    n_real_unique = df['Cleaned_EA'].n_unique()
    if n_real_unique != n_unique:
        ej = (df['Cleaned_EA'].to_pandas().value_counts().index[:5])
        st.warning("⚠️ Hay EAs que solo se distinguen por mayúsculas/minúsculas o espacios finales/iniciales. "
                   "Ejemplos: " + ", ".join(ej))

    st.sidebar.header("Filtros")
    fecha_min = df['Close Time'].min().date()
    fecha_max = df['Close Time'].max().date()
    fecha_desde, fecha_hasta = st.sidebar.date_input(
        'Rango de fechas (Close Time)',
        value=[fecha_min, fecha_max],
        min_value=fecha_min, max_value=fecha_max,
        help="Filtra por la fecha de cierre de las operaciones."
    )
    if isinstance(fecha_desde, list):
        fecha_desde, fecha_hasta = fecha_desde
    if fecha_desde > fecha_hasta:
        st.warning("La fecha 'Desde' no puede ser mayor que 'Hasta'. Corrígelo en el filtro.")
        st.stop()
    multiplicador = st.sidebar.number_input(
        "Multiplicador", value=1.0, min_value=0.0, step=0.1,
        help="Úsalo para ajustar el tamaño del lote o el profit (por ejemplo, según el capital)."
    )
    filtro_texto = st.sidebar.text_input(
        "Buscar texto en nombre de EA:", "",
        help="Separa varios términos con punto y coma ';' o coma ',' para buscar varios EAs a la vez. Ejemplo: grid,scalp,trend"
    )
    profit_min = st.sidebar.number_input(
        "Profit total mínimo", value=-99999.0,
        help="Oculta EAs con profit menor a este valor.")
    dd_maximo_max = st.sidebar.number_input("DD máximo absoluto límite", value=9999999.0, help="Define el drawdown máximo permitido (negativo, peak-to-trough).")
    ocultar_no_selec = st.sidebar.checkbox("Ocultar EAs no seleccionados manualmente", value=False)

    mask_fechas = (df['Close Time'].dt.date() >= fecha_desde) & (df['Close Time'].dt.date() <= fecha_hasta)
    df_f = df.filter(mask_fechas)
    if filtro_texto.strip():
        terminos = [t.strip() for t in filtro_texto.replace(",", ";").split(";") if t.strip()]
        if terminos:
            mask = df_f.select(
                pl.col('Cleaned_EA')
                  .str.strip_chars()
                  .str.to_lowercase()
                  .map_elements(lambda x: any(t.lower() in x for t in terminos), return_dtype=pl.Boolean)
                  .alias('filtro')
            )['filtro']
            df_f = df_f.filter(mask)
    if df_f.is_empty():
        st.warning("No hay datos que coincidan con el filtro actual.")
        st.stop()
    resumen, meses_disponibles = calcular_resumen(df_f, multiplicador)
    meses_cols = [col for col in resumen.columns if col.startswith('Profit_20')]
    resumen = resumen.filter(
        (pl.col('Profit_Total') >= profit_min) &
        (pl.col('DD_Max_Curva') >= -dd_maximo_max)
    )

    st.markdown("---")
    st.markdown("#### DD máximo global (portafolio de todas las EAs seleccionadas combinadas, en el periodo)")
    df_portafolio = df_f.filter(pl.col('Cleaned_EA').is_in(resumen['Cleaned_EA']))
    df_portafolio = df_portafolio.sort('Close Time')
    df_portafolio = df_portafolio.with_columns(
        (pl.col('Profit') * multiplicador).alias('Profit_neto'),
        (pl.col('Commission') * multiplicador).alias('Commission_multi'),
        (pl.col('Taxes') * multiplicador).alias('Taxes_multi'),
        (pl.col('Swap') * multiplicador).alias('Swap_multi')
    )
    df_portafolio = df_portafolio.with_columns(
        pl.col('Profit_neto').cum_sum().alias('Saldo_Acumulado')
    )
    saldo = df_portafolio['Saldo_Acumulado'].to_numpy()
    peak = np.maximum.accumulate(saldo)
    dd = saldo - peak
    dd_max_portafolio = dd.min() if dd.size > 0 else np.nan

    st.metric("DD máximo global (portafolio)", f"{dd_max_portafolio:,.2f}")
    mostrar_kpis(resumen, dd_max_portafolio, df_portafolio, multiplicador)

    fig = px.line(df_portafolio.to_pandas(), x='Close Time', y='Saldo_Acumulado', title="Curva de saldo combinada seleccionada")
    st.plotly_chart(fig, use_container_width=True)
    figdd = px.line(df_portafolio.to_pandas(), x='Close Time', y=dd, title="Drawdown global en el tiempo")
    st.plotly_chart(figdd, use_container_width=True)
    st.dataframe(resumen.to_pandas().set_index('Cleaned_EA'), use_container_width=True)

    # =========== KPIs DETALLADOS Y EXPORTACIÓN EXCEL ===========
    rp = resumen.to_pandas().set_index('Cleaned_EA')
    df_det = df_f.to_pandas()
    profit_neto_det = (df_det['Profit'] * multiplicador)
    n_total = len(df_det)
    n_ganadoras = (profit_neto_det > 0).sum()
    n_perdedoras = (profit_neto_det < 0).sum()
    media_ganadoras = round(profit_neto_det[profit_neto_det > 0].mean(), 2)
    media_perdedoras = round(profit_neto_det[profit_neto_det < 0].mean(), 2)
    n_tp = ((profit_neto_det > 0) & (np.abs(df_det['T / P'] - df_det['Close Price']) < 1e-3)).sum()
    n_sl = ((profit_neto_det < 0) & (profit_neto_det <= -180)).sum()
    sum_ganadoras = profit_neto_det[profit_neto_det > 0].sum()
    sum_perdedoras = profit_neto_det[profit_neto_det < 0].sum()
    profit_factor = round(sum_ganadoras / abs(sum_perdedoras), 2) if sum_perdedoras != 0 else np.nan
    winning_rate = round((n_ganadoras / n_total) * 100, 2) if n_total > 0 else 0
    ret_dd_global = round(rp['Profit_Total'].sum() / abs(dd_max_portafolio),2) if dd_max_portafolio != 0 else np.nan

    kpi_dict = {
        "Nº total operaciones": n_total,
        "Nº operaciones ganadoras": n_ganadoras,
        "Nº operaciones perdedoras": n_perdedoras,
        "Winning %": winning_rate,
        "Media operaciones ganadoras": media_ganadoras,
        "Media operaciones perdedoras": media_perdedoras,
        "Nº de TP (profit>0 y precio igual TP)": n_tp,
        "Nº de SL (profit<=-180)": n_sl,
        "Profit Factor global": profit_factor,
        "Ret/DD global": ret_dd_global,
    }

    with st.expander("KPIs detallados de todos los trades (para Excel):"):
        for k, v in kpi_dict.items():
            st.write(f"**{k}:** {v}")

    df_kpi_export = pd.DataFrame(
        [
            ["[KPI] "+k, v] + [np.nan]*(len(rp.columns)-1)
            for k, v in kpi_dict.items()
        ],
        columns=["Cleaned_EA"] + list(rp.columns)
    )

    export_excel = pd.concat([
        rp.reset_index(),
        df_kpi_export
    ], ignore_index=True)

    descargar_dataframe(pl.from_pandas(export_excel), "resumen_EAs.xlsx", "Descargar tabla resumen en Excel")
    descargar_dataframe(df_f, "Trades_EAs_filtrado.xlsx", "Descargar Todos Trades en Excel")
else:
    st.info("Por favor, carga tus archivos Excel o CSV primero.")