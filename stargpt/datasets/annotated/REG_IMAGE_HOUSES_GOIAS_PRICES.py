from os.path import exists, join
from typing import Optional

from pandas import DataFrame

from tabstar2.utils.datasets import load_csv
from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: carloseduardogo/apartment-prices-in-goinia-gois-brazil/
====
Examples: 9833
====
URL: https://www.kaggle.com/carloseduardogo/apartment-prices-in-goinia-gois-brazil
====
Description: 
Apartment Prices in Goiânia, Goiás, Brazil
Real data on characteristics and prices of apartments in Goiânia, Goiás, Brazil.

About Dataset
Data Collection
The data was collected by web scrapping (using Selenium) from the website vivareal.com.br. Please note that this was done solely for educational purposes. More details regarding the data collection process can be found in my github repository.

Data Available
The available data includes characteristics of the apartments, such as price, condominium price, IPTU value, number of rooms, number of bathrooms, floor level, number of parking spaces, total area of ​​the apartment, main image of the apartment advertisement, complete description of the advertiser and binary columns indicating the presence or absence of certain amenities. It is a unique opportunity to apply regression models and data cleaning techniques to real data, including images and complete descriptions, of apartment prices!

Data Science Project Example
In my github repository, there is an example of applying machine learning to predict apartment prices in Goiânia, using the dataset available here. At the end of the project, an average percentage error of close to 14% was obtained in the data separated for testing. The model was deployed via streamlit, and you can access the web application by clicking here. Note that you can consult the github repository if you need help with data preprocessing or similar details.
====
Target Variable: price (float64, 1784 distinct): ['450.0', '550.0', '650.0', '350.0', '400.0', '600.0', '460.0', '750.0', '850.0', '1100.0']
====
Features:

tag_card (object, 5 distinct, 63.3% missing): ['Destaque', 'Super Destaque', 'Em construção', 'Pronto para morar', 'Na planta']
business_type (object, 2 distinct): ['Venda', 'A partir de']
address (object, 3878 distinct): ['Setor Bueno, Goiânia - GO', 'Setor Oeste, Goiânia - GO', 'Setor Marista, Goiânia - GO', 'Parque Amazônia, Goiânia - GO', 'Rua 56, 79 - Jardim Goiás, Goiânia - GO', 'Rua T 36 - Setor Bueno, Goiânia - GO', 'Jardim Goiás, Goiânia - GO', 'Setor Pedro Ludovico, Goiânia - GO', 'Rua T 30 - Setor Bueno, Goiânia - GO', 'Rua T 37 - Setor Bueno, Goiânia - GO']
floorSize (object, 348 distinct): ['64 m²', '76 m²', '58 m²', '75 m²', '70 m²', '60 m²', '68 m²', '74 m²', '56 m²', '114 m²']
numberOfRooms (object, 10 distinct, 0.0% missing): ['3 quartos', '2 quartos', '4 quartos', '1 quarto', '5 quartos', '6 quartos', '2 - 3 quartos', '10 quartos', '7 quartos', '11 quartos']
numberOfBathroomsTotal (object, 13 distinct, 0.0% missing): ['2 banheiros', '4 banheiros', '3 banheiros', '1 banheiro', '5 banheiros', '6 banheiros', '7 banheiros', '2 - 3 banheiros', '9 banheiros', '11 banheiros']
numberOfParkingSpaces (object, 18 distinct, 2.7% missing): ['2 vagas', '1 vaga', '3 vagas', '4 vagas', '5 vagas', '1 - 2 vagas', '6 vagas', '7 vagas', '12 vagas', '10 vagas']
floorLevel (object, 54 distinct, 54.3% missing): ['2 andar', '3 andar', '5 andar', '6 andar', '4 andar', '9 andar', '10 andar', '8 andar', '7 andar', '1 andar']
pool (object, 1 distinct, 49.7% missing): ['Piscina']
elevator (object, 1 distinct, 49.3% missing): ['Elevador']
barbecueGrill (object, 1 distinct, 53.4% missing): ['Churrasqueira']
gatedCommunity (object, 1 distinct, 64.3% missing): ['Condomínio fechado']
gym (object, 1 distinct, 48.9% missing): ['Academia']
gourmetSpace (object, 1 distinct, 58.3% missing): ['Espaço gourmet']
garden (object, 1 distinct, 74.9% missing): ['Jardim']
playground (object, 1 distinct, 50.8% missing): ['Playground']
partyHall (object, 1 distinct, 38.6% missing): ['Salão de festas']
laundry (object, 1 distinct, 92.0% missing): ['Lavanderia']
disabledAccess (object, 1 distinct, 82.7% missing): ["Acesso para PCD's"]
coworking (object, 1 distinct, 94.2% missing): ['Coworking']
electronicGate (object, 1 distinct, 75.6% missing): ['Portão eletrônico']
sauna (object, 1 distinct, 73.1% missing): ['Sauna']
spa (object, 1 distinct, 88.7% missing): ['Spa']
complete_description (object, 6916 distinct): ['Excelente apartamento no Setor Parque Amazônia, lançamento da CMO Construtora.\r\n\r\nO Residencial Solar Amazônia fica localizado do lado da Praça da CEMACO no Parque Amazônia, contando com Bancos, farmácias, Panificadora Cerrado Pães, Posto de Gasolina e uma vasta rede de Comercio. A 2 minutos do colégio WR Junior, a 3 minutos do Buriti Shopping e Atacarejo Assai, a 5 minutos do Parque Cascavel. Pertinho do Setor Bueno, Setor Bela Vista, Setor Marista e acesso fácil ao Jardim América. O empreendimento conta com apartamento de 2 e 3 quartos sendo 2 Quartos com 1 suite de 64 m² e o de 3 Quartos com 1 suite 76 e 83 m², todos com excelente acabamento em piso Porcelanato 60X60, granito Arabesco e ponto de ar-condicionado Split na suite.\r\n\r\nCondomínio com lazer completo, piscina adulto e infantil, academia, salão de festa, churrasqueira, brinquedoteca, playground e portaria com pulmão de segurança ao acesso de visitantes, trazendo segurança e conforto para sua família.', "O Uptown estará no melhor do Jardim Europa entre avenidas largas como a Milão e a Veneza, que dão acesso a várias regiões da cidade por meio de importantes avenidas como a T-9 e a T-7. Próximo ao novo Parque Bernardo Élis, e a um rico polo gastronômico, o que não faltam são opções que oferecem mais conforto e agilidade ao seu dia-a-dia.\r\nMCDonald's, Kanpai", 'Apartamento Versátil e Confortável à Venda no Residencial Xavier Jr, Setor Bueno - R$ 490.000\r\n\r\nDescrição do Imóvel:\r\n\r\n3 Quartos: Amplos e bem iluminados, oferecendo conforto e privacidade para toda a família.\r\nEscritório: Espaço ideal para home office ou estudo.\r\n3 Banheiros: Incluindo um lavabo, proporcionando comodidade para os visitantes.\r\nCozinha Isolada: Ambiente prático e funcional para suas atividades culinárias.\r\nSala de Jantar Separada: Perfeita para refeições em família e momentos especiais.\r\nSala de Estar em Dois Ambientes: Espaçosa e ideal para relaxamento e entretenimento.\r\nAmplo Closet: Espaço adicional para armazenamento e organização.\r\nAr-Condicionado: Conforto térmico garantido em todas as estações do ano.\r\nDestaques:\r\n\r\nLocalização: Situado no Residencial Xavier Jr, no prestigiado Setor Bueno, próximo a diversas comodidades e serviços.\r\nConforto e Versatilidade: Ideal para famílias que buscam um espaço bem planejado e funcional.\r\nAmbientes Amplos e Bem Distribuídos: Proporcionando conforto e praticidade para o seu dia a dia.\r\nValor de Investimento: R$ 490.000\r\n\r\nNão perca a oportunidade de viver em um apartamento que combina conforto e conveniência! Entre em contato agora mesmo para mais informações e agende uma visita.', 'OPORTUNIDADE ÚNICA! SEU NOVO LAR NO PARQUE AMAZÔNIA!\r\n\r\nApartamento dos Sonhos - Pronto para Morar!\r\n\r\nSeja bem-vindo(a) à sua nova vida de luxo no coração do Parque Amazônia! Este espetacular apartamento de 3 quartos, SENDO 1 SUÍTE, espera por você no andar ALTO, proporcionando vistas deslumbrantes da cidade. Uma oportunidade única de viver com conforto, elegância e praticidade!\r\n\r\nPrincipais Características:\r\n\r\n3 Quartos, SENDO 1 SUÍTE: Espaço generoso para toda a família, com um refúgio privativo para momentos especiais.\r\n\r\nAndar Alto: Desfrute de vistas panorâmicas incríveis da cidade, um convite diário para admirar o pôr do sol e as luzes da metrópole.\r\n\r\n2 Vagas de Garagem: Conveniência e segurança para seus veículos, proporcionando praticidade no seu dia a dia.\r\n\r\n79 m² de Área Útil: Um espaço bem distribuído para que cada canto do seu lar seja aproveitado da melhor forma possível.\r\n\r\nRico em Armários: Armários planejados de alta qualidade em todos os cômodos, garantindo organização e beleza aos seus ambientes.\r\n\r\nValor Imperdível: R$510.000,00\r\n\r\nAgende já sua visita e garanta seu lugar neste pedaço do paraíso!\r\n\r\nBenefícios Exclusivos:\r\n\r\nPronto para Morar: Mude-se imediatamente para o seu novo lar, sem preocupações com reformas.\r\n\r\nAcabamentos de Alto Padrão: Detalhes cuidadosamente escolhidos para agregar sofisticação aos espaços.\r\n\r\nLocalização Privilegiada: No coração do Parque Amazônia, próximo a comércios, escolas e facilidades para seu dia a dia.\r\n\r\nÁrea de Lazer Completa: Desfrute de momentos de lazer sem sair de casa, com opções como piscina, academia e salão de festas.\r\n\r\nSegurança 24 horas: Tranquilidade para você e sua família.\r\n\r\nNão perca tempo! Este é o investimento que você esperava para viver com qualidade e conforto.\r\n\r\nAgende sua visita agora mesmo e faça deste apartamento o seu novo lar!', 'Seu Novo Refúgio no Eldorado! Apartamento no Água Marinha - Uma Oportunidade Imperdível!\r\n\r\nDesfrute do conforto e elegância neste espetacular apartamento no 14º andar, com uma área de 93,31m² totalmente revestida em porcelanato. Com uma localização privilegiada no Eldorado, este é o lar perfeito para você!\r\n\r\nCaracterísticas do Apartamento:\r\n\r\n3 Suítes\r\nCozinha Fechada com Rico Armário\r\nEscritório para Trabalho ou Estudo\r\nSala para 2 Ambientes\r\nAr Condicionado para o seu Conforto\r\n2 Vagas de Garagem Individuais\r\n\r\nLazer Completo no Condomínio Clube:\r\n\r\nPiscina Adulto e Infantil\r\nSauna\r\nBrinquedoteca\r\nSalão de Festas\r\nBar da Piscina\r\nPlayground\r\nQuadra Poliesportiva\r\nChurrasqueira\r\nÁrea de Convivência\r\n\r\nAlém disso, você terá uma vista privilegiada do penúltimo andar e estará a apenas 1 minuto do Shopping Plaza Doro. A praticidade aliada ao lazer e conforto!\r\n\r\nAceita Financiamento!\r\n\r\nPor apenas R$ 549.900,00, você pode garantir esse incrível apartamento e viver a vida que sempre sonhou. Não perca tempo, agende já sua visita e faça deste lugar o seu novo lar! Entre em contato agora mesmo e não deixe escapar essa oportunidade única.', 'Seu Novo Estilo de Vida no Residencial Vistta Buriti, Parque Amazônia!\r\n\r\nDescubra o conforto e praticidade neste incrível apartamento no 6º andar, com 58 m² de puro charme no Residencial Vistta Buriti. Com 2 quartos, sendo um suíte, e 1 vaga de garagem no subsolo, este imóvel oferece o ambiente perfeito para quem busca comodidade e estilo.\r\n\r\nDetalhes do Apartamento:\r\n\r\n2 Quartos, 1 Suíte\r\nBanheiro com Box na Suíte\r\nCozinha com Armários e Bancada de Quartzo Branco\r\n1 Vaga de Garagem no Subsolo\r\nCortinas na Suíte e na Sala\r\nAr Condicionado na Suíte\r\nItens Inclusos - PORTEIRA FECHADA:\r\n\r\nFogão de Indução 4 Bocas\r\nForno\r\nCoifa\r\nGeladeira\r\nSofá\r\nMesa com 4 Cadeiras\r\n1 Cama de Casal\r\nCondomínio e Lazer:\r\n\r\nCondomínio: R$400,00\r\nLazer na Cobertura\r\nValor Atrativo: R$439.000,00\r\n\r\nDesfrute de um estilo de vida moderno e prático, onde tudo que você precisa está à sua disposição. O Residencial Vistta Buriti oferece não apenas um lar, mas uma experiência de moradia completa.\r\n\r\nAgende sua visita agora mesmo e venha se encantar com esse apartamento cheio de charme e personalidade. Entre em contato para mais informações e faça deste imóvel o seu novo lar!', 'Na melhor localização do Setor Bueno, a 300 Metros do Parque Vaca Brava e Goiânia Shopping.\r\n\r\nO Ello Vaca Brava e composto por apartamentos de 3 Suítes de 103m², 114m² e 138m². Planta com integração perfeita, fino acabamento, casal com closet, esquadrias balcão na suíte casal, possibilitando melhor iluminação e ventilação natural, balcão privativo com área para jardim nos aptos tipo de 114m², cozinha possibilitando integração com sala, planta baixa aptos tipo sem pilares internos, ponto ar condicionado em todas suítes e na sala, todos os banheiros das suítes ventilados naturalmente, plantas flexíveis com possibilidade de personalização com a construtora durante a construção.\r\n\r\nArea de lazer entregue mobiliado e decorado pela arquiteta Adriana Mundim:\r\n\r\nAr-condicionado entregue na guarita, academia, salão de festas, sport lounge e co-study e coworking;\r\nEspaço gourmet, salão de festas e sports lounge entregues com utensílios domésticos, (pratos, talheres e copos);\r\nAcademia com 98,47m² com área de pilates e acesso por biometria;\r\nSalão de festas e lounge festas com 130,35m². Lounge do salão com espaço de contemplação e churrasqueira gourmet;\r\nEspaço gourmet independente com churrasqueira e lounge contemplação;\r\nSports Lounge e jogos para adultos;\r\nAmplo lazer infantil com playground de 33,12m², piscina infantil, brinquedoteca multifuncional com 27,84m²;\r\nEntrega de 1 speed dome;\r\nPrevisão de portaria remota;\r\nParceria com a TECNOSEG para projeto de segurança;\r\nAcesso biometrico na portaria;\r\nAcesso biometrico para a entrada e saida de veiculos.', 'Vendo Espaçoso Apartamento de 4 Quartos com Área de Lazer Completa\r\n\r\nVocê está em busca de um lar que ofereça conforto, espaço e comodidade? Apresentamos o apartamento dos seus sonhos! Esta é uma oportunidade única para adquirir um apartamento de alto padrão com uma infinidade de comodidades. Confira os detalhes deste incrível imóvel:\r\n\r\nCaracterísticas do Apartamento:\r\n\r\n4 Quartos\r\n1 Suíte Master: Com banheiro privativo e detalhes luxuosos.\r\n1 Suíte Plena: Com ótimo espaço e privacidade.\r\n2 Suítes Americanas: Ambas com acesso ao mesmo banheiro, proporcionando praticidade.\r\nUnidade Exclusiva: Este é o único apartamento disponível com 3 garagens individuais, ideal para famílias com mais de um veículo ou colecionadores de carros.\r\nNascente: Desfrute de luz solar pela manhã e ambientes bem iluminados.\r\nVaranda: Um espaço aconchegante para relaxar e apreciar a vista.\r\nÁrea de Serviço: Prático espaço dedicado às tarefas do dia a dia.\r\nArmários: Cozinha e quartos com armários planejados, oferecendo organização e estilo.\r\nChurrasqueira: Perfeito para reunir amigos e familiares e desfrutar de deliciosos churrascos.\r\nMobiliado: O apartamento é vendido com todos os móveis, tornando a mudança ainda mais fácil e rápida.\r\nDetalhes da Área de Lazer do Condomínio:\r\n\r\nAcademia Completa: Mantenha-se em forma com equipamentos modernos e espaços bem projetados.\r\nPiscinas Aquecidas: Relaxe e divirta-se em piscinas aquecidas durante todo o ano.\r\nQuadra Esportiva: Desfrute de momentos esportivos e de lazer em uma quadra bem cuidada.\r\nSauna: Renove suas energias com uma relaxante sessão de sauna.\r\nSalão de Festas com Ar Condicionado: Amplo espaço para celebrações memoráveis, equipado com todos os utensílios necessários para festas.\r\nÁrea de Lazer Completa: Uma gama de opções para o entretenimento de toda a família.\r\nDetalhes do Condomínio:\r\n\r\nCondomínio Fechado: Privacidade e segurança para os moradores.\r\nElevador: Acesso facilitado a todos os andares.\r\nPermitido Animais: Se você tem um companheiro de quatro patas, ele será bem-vindo.\r\nPortaria e Segurança 24h: Tranquilidade e proteção para você e sua família.\r\nLocalização:\r\nO apartamento está situado em uma região privilegiada, com fácil acesso a serviços, comércio e vias importantes da cidade. Desfrute de todas as comodidades que essa localização exclusiva oferece.\r\n\r\nNão perca esta oportunidade de adquirir um apartamento verdadeiramente único, com espaço, lazer e comodidades diferenciadas. O imóvel está quitado e pronto para transferência imediata. Agende uma visita e encante-se!', 'A NEWINC Incorporadora acaba de lançar no Setor Eldorado um projeto grandioso. O Terra Mundi Eldorado terá um dos maiores complexos de lazer da região, para atender a necessidade de todos os tipos de pessoas e famílias. A Incorporadora e Construtora adquiriu o projeto Invent da Brookfiled Incorporações e trouxe as premissas Terra Mundi para torná-lo singular.\r\n\r\nCom um novo conceito eco-sustentável, novos diferenciais e detalhes minunciosamente pensados para fazer do Terra Mundi Eldorado o empreendimento para todos os momentos da sua vida.\r\n\r\nApartamento 3 Quartos 3 Suítes e área privativa de 88,11m² , com alto padrão de acabamento: piso Porcelanato polido em todo o apartamento, varanda gourmet com ponto para churrasqueira à gás e ponto hidráulico\r\n\r\nClimatização de quartos e salas com aparelhos de ar condicionado já instalados(os aparelhos já serão entregues pela construtora em todo apartamento)\r\n\r\nÁgua nos chuveiros com sistema de aquecimento solar e ponto de água de reuso, benefícios eco-sustentáveis que reflete numa enorme economia pessoal de água e energia e também no valor do condomínio.\r\n\r\nAlém disso, NEWINC é responsável pela construção do Invent Clube, que é anexo aos condomínios Invent Max, Joy e, agora, ao Terra Mundi Eldorado.\r\nO Invent Clube será entregue simultaneamente com a última torre do Terra Mundi Eldorado.', 'Apartamento reformado à venda no condomínio fechado Recanto Praças 2 no setor Negrão de Lima.\r\n\r\n2 quartos\r\nBanheiro social\r\nCozinha com bancada em granito no formato “u” com fogão cooktop novo\r\n64m²\r\nPiso em porcelanato polido em todo apartamento\r\nBanheiro reformado\r\nO apartamento está em fase final de reforma, e será entregue finalizado: com pintara, fechamento de vidro blindex na varanda e limpeza pós obra.\r\nPrédio possui elevador\r\n\r\nAmpla varanda com churrasqueira à carvão (varanda gourmet).\r\n1 vaga de garagem\r\n\r\nO condomínio fechado possui piscina adulto aquecida e piscina infantil, playground, quadra poliesportiva, campo de futebol gramado, campo de futebol de areia, salão de festas equipado, quiosques com churrasqueiras, mesas e cadeiras, ampla área verde além de um bosque privativo com pista de caminhada que enriquece esse ar bucólico que tanto amamos no Recanto Praças. Portaria com segurança 24horas.\r\n\r\nLocalizado próximo aos setores: Centro, Vila Nova, Setor Jaó, Universitário.\r\n\r\n*Aceita financiamento.\r\n\r\n___________________________\r\nOPORTUNIDADE: de R$ 319.000,00 por R$ 305.000,00.\r\nValor do condomínio: R$350\r\n___________________________']
petsAllowed (object, 1 distinct, 64.8% missing): ['Animais permitidos']
airConditioning (object, 1 distinct, 71.7% missing): ['Ar condicionado']
gourmetBalcony (object, 1 distinct, 78.7% missing): ['Varanda gourmet']
sportsCourt (object, 1 distinct, 64.4% missing): ['Quadra poliesportiva']
bicyclesPlace (object, 1 distinct, 86.6% missing): ['Bicicletário']
concierge24h (object, 1 distinct, 74.1% missing): ['Portaria 24h']
americanKitchen (object, 1 distinct, 87.9% missing): ['Cozinha americana']
balcony (object, 1 distinct, 80.5% missing): ['Varanda']
intercom (object, 1 distinct, 74.1% missing): ['Interfone']
serviceArea (object, 1 distinct, 67.2% missing): ['Área de serviço']
largeKitchen (object, 1 distinct, 97.6% missing): ['Cozinha grande']
furnished (object, 1 distinct, 90.4% missing): ['Mobiliado']
kitchen (object, 1 distinct, 78.2% missing): ['Cozinha']
adultGameRoom (object, 1 distinct, 78.2% missing): ['Salão de jogos']
deposit (object, 1 distinct, 96.6% missing): ['Depósito']
watchman (object, 1 distinct, 93.1% missing): ['Porteiro']
security24Hours (object, 1 distinct, 90.2% missing): ['Segurança 24 horas']
bedroomWardrobe (object, 1 distinct, 95.8% missing): ['Armário embutido no quarto']
internetAccess (object, 1 distinct, 93.2% missing): ['Acesso à internet']
closet (object, 1 distinct, 87.5% missing): ['Closet']
gourmetKitchen (object, 1 distinct, 98.4% missing): ['Cozinha gourmet']
electricGenerator (object, 1 distinct, 95.6% missing): ['Gerador']
alarmSystem (object, 1 distinct, 96.0% missing): ['Alarme']
toysPlace (object, 1 distinct, 96.1% missing): ['Brinquedoteca']
cinema (object, 1 distinct, 96.4% missing): ['Cinema']
heating (object, 1 distinct, 98.2% missing): ['Aquecimento']
builtinWardrobe (object, 1 distinct, 88.4% missing): ['Armário embutido']
kitchenCabinets (object, 1 distinct, 89.4% missing): ['Armários na cozinha']
safetyCircuit (object, 1 distinct, 91.5% missing): ['Circuito de segurança']
garage (object, 1 distinct, 82.1% missing): ['Garagem']
tennisCourt (object, 1 distinct, 97.2% missing): ['Quadra de tênis']
greenSpace (object, 1 distinct, 96.4% missing): ['Área verde']
squash (object, 1 distinct, 97.5% missing): ['Squash']
fireplace (object, 1 distinct, 99.4% missing): ['Lareira']
homeOffice (object, 1 distinct, 96.5% missing): ['Escritório']
bathroomCabinets (object, 1 distinct, 96.4% missing): ['Armário no banheiro']
gasShower (object, 1 distinct, 99.8% missing): ['Chuveiro a gás']
blindexBox (object, 1 distinct, 97.1% missing): ['Box blindex']
reception (object, 1 distinct, 94.3% missing): ['Recepção']
cableTv (object, 1 distinct, 98.1% missing): ['TV à cabo']
bathtub (object, 1 distinct, 99.5% missing): ['Banheira']
privatePool (object, 1 distinct, 99.8% missing): ['Piscina privativa']
backyard (object, 1 distinct, 99.2% missing): ['Quintal']
grass (object, 1 distinct, 99.4% missing): ['Gramado']
largeWindow (object, 1 distinct, 98.2% missing): ['Janelas grandes']
naturalVentilation (object, 1 distinct, 97.9% missing): ['Ventilação natural']
cooker (object, 1 distinct, 99.1% missing): ['Fogão']
hikingTrail (object, 1 distinct, 99.7% missing): ['Trilha para caminhada']
panoramicView (object, 1 distinct, 98.5% missing): ['Vista panorâmica']
massage (object, 1 distinct, 99.8% missing): ['Massagem']
freezer (object, 1 distinct, 100.0% missing): ['Freezer']
department pic (object, 9832 distinct, 0.0% missing): ['62750619205425081544300816117885861273.png', '302678336848826931249389939487192177877.png', '130359292211767838527233708900928593543.png', '191007733490619722070589252030900058169.png', '200978065182550317495009592604134775241.png', '9796372657926373412148234085533873844.png', '263139753630466744298302557714647552471.png', '303356502250666574948619601235945089970.png', '44848129587218883434001225188777021263.png', '51558828263966667869360010860225643131.png']
'''

def load_df(dir_path: str) -> DataFrame:    
    df = load_csv(dir_path, "data.csv", sep="|")
    df[IMAGE_FEATURE_NAME] = df['id'].apply(lambda id: _get_img(id, dir_path))
    return df

IMAGE_FEATURE_NAME = "department pic"


def _get_img(dep_id: int, dir_path: str) -> Optional[str]:
    img_folder = join(dir_path, IMAGE_FOLDER)
    img_filename = f"{dep_id}.png"
    if not exists(join(img_folder, img_filename)):
        return None
    return img_filename

def _parse_brazilian_currency(price: str) -> Optional[float]:
    # R$ 1.000.000, R$ 949.900
    if price in ['Sob consulta']:
        return None
    price = price.replace('R$', '').strip()
    if price.count('.') == 2:
        # replace the first
        price = price.replace('.', '', 1)
    return float(price)

def _parse_floorsize(floorsize: str) -> float | None:
    # '100 m²'
    floorsize = floorsize.replace(' m²', '').strip()
    try:
        return float(floorsize)
    except ValueError:
        return None

CONTEXT = ""
TARGET = CuratedTarget(raw_name='price', task_type=SupervisedTask.REGRESSION, processing_func=_parse_brazilian_currency)
COLS_TO_DROP = ['id',
                # Leakage?
                'condo_fee', 'iptu']
TEXT_FEATURES = [CuratedFeature(raw_name=f, feat_type=FeatureType.TEXT) for f in ['address', 'complete_description', 'iptu']]
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE),
            CuratedFeature(raw_name='floorSize', feat_type=FeatureType.NUMERIC, processing_func=_parse_floorsize),
            ] + TEXT_FEATURES
IMAGE_FOLDER = "imgs/imgs"
LOADING_FUNC = load_df
