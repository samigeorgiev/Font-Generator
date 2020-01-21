import React, { Component } from 'react';

import styles from './index.module.css';

class Saved extends Component {
    state = {
        fonts: null
    };

    componentDidMount() {
        const url = process.env.REACT_APP_BASE_URL + process.env.REACT_APP_GET_SAVED_FONTS_PATH;
        // fetch(url).then(data => data.json()).then(fonts => {
        //     this.setState({
        //         fonts: fonts
        //     });
        // }).catch(e => console.log(e));
            this.setState({
                fonts: [
                    { heading: 'dfdfgdfg', body: 'dsfgfs' },
                    { heading: 'dfdfgdfg', body: 'dsfgfs' },
                    { heading: 'dfdfgdfg', body: 'dsfgfs' },
                    { heading: 'dfdfgdfg', body: 'dsfgfs' }
                ]
            });
    }

    render() {
        return (
            <main className={styles.Saved}>
                {this.state.fonts && this.state.fonts.map(font => (
                    <div>
                        <h2>Heading font: {font.heading}</h2>
                        <h3>Body font: {font.body}</h3>
                    </div>
                ))}
            </main>
        );
    }
}

export default Saved;