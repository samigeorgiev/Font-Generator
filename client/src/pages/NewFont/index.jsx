import React, { Component } from 'react';

import webFontLoader from 'webfontloader';

import Slider from 'components/Slider';

import styles from './index.module.css';

class NewFont extends Component {
    state = {
        properties: {
            contrast: {
                prev: 0,
                cur: 0
            },
            thickness: {
                prev: 0,
                cur: 0
            }
        },
        fonts: {
            heading: null,
            body: null
        },
        lastRequestController: null
    };

    propertyHandler = async (property, e) => { // TODO TEST
        this.state.lastRequestController && this.state.lastRequestController.abort();
        const controller = new AbortController();
        const signal = controller.signal;
        await this.setState({
            properties: {
                ...this.state.properties,
                [property]: {
                    ...this.state.properties[property],
                    cur: e.target.value
                }
            },
            lastRequestController: controller
        });
        const url = process.env.REACT_APP_BASE_URL + process.env.REACT_APP_NEW_FONT_PATH;
        const options = {
            method: 'POST',
            headers: {
                'Authorization': this.props.user?.token,
                'Content-type': 'application/json'
            },
            body: JSON.stringify({
                deltaContrast: this.state.properties.contrast.cur - this.state.properties.contrast.prev,
                deltaThickness: this.state.properties.thickness.cur - this.state.properties.thickness.prev,
                fonts: this.state.fonts
            }),
            signal
        };
        console.log(options);
        return;
        fetch(url, options).then(data => data.json()).then(fonts => {
            this.changeFonts({
                heading: 'Amarante',
                body: 'Amarante'
            });
        }).catch(err => {
            console.log(err);
        });
    };

    changeFonts = async ({heading, body}) => {
        webFontLoader.load({
            google: {
                families: [heading, body]
            },
        });
        this.setState({
            fonts: {
                heading: heading,
                body: body
            }
        });
    };

    componentDidMount() { // TODO: TEST
        const url = process.env.REACT_APP_BASE_URL + process.env.REACT_APP_RECOMMEND_PATH;
        this.changeFonts({
            heading: 'Amarante',
            body: 'Amarante'
        });
        return;
        fetch(url, {
            headers: {
                'Authorization': this.props.user?.token
            }
        }).then(data => data.json()).then(fonts => {
            this.setState({
                fonts: {
                    heading: fonts.heading,
                    body: fonts.body
                }
            });
        }).catch(err => console.log(err));
    }

    render() {
        return (
            <main className={styles.NewFont}>
                {Object.entries(this.state.properties).map(([property, value]) => (
                    <Slider
                        key={property}
                        min="-1"
                        max="1"
                        step="0.01"
                        name={property}
                        value={value.cur}
                        change={e => this.propertyHandler(property, e)}
                    />
                ))}
                <p style={{fontFamily: this.state.fonts.body}}>Test</p>
            </main>
        );
    }
}

export default NewFont;